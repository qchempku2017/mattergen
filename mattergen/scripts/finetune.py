# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import logging
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Tuple

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.cli import SaveConfigCallback

from mattergen.diffusion.config import Config
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.common.utils.globals import MODELS_PROJECT_ROOT, get_device
from mattergen.diffusion.run import AddConfigCallback, SimpleParser, maybe_instantiate

logger = logging.getLogger(__name__)


def init_adapter_lightningmodule_from_pretrained(
    adapter_config: DictConfig, lightning_module_config: DictConfig
) -> Tuple[pl.LightningModule, DictConfig]:

    if adapter_config.model_path is not None:
        if adapter_config.pretrained_name is not None:
            logger.warning(
                "pretrained_name is provided, but will be ignored since model_path is also provided."
            )
        model_path = Path(hydra.utils.to_absolute_path(adapter_config.model_path))
        ckpt_info = MatterGenCheckpointInfo(model_path, adapter_config.load_epoch)
    elif adapter_config.pretrained_name is not None:
        assert (
            adapter_config.model_path is None
        ), "model_path must be None when pretrained_name is provided."
        ckpt_info = MatterGenCheckpointInfo.from_hf_hub(adapter_config.pretrained_name)

    ckpt_path = ckpt_info.checkpoint_path

    version_root_path = Path(ckpt_path).relative_to(ckpt_info.model_path).parents[1]
    config_path = ckpt_info.model_path / version_root_path

    # load pretrained model config.
    if (config_path / "config.yaml").exists():
        pretrained_config_path = config_path
    else:
        pretrained_config_path = config_path.parent.parent

    # global hydra already initialized with @hydra.main
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    with hydra.initialize_config_dir(str(pretrained_config_path.absolute()), version_base="1.1"):
        pretrained_config = hydra.compose(config_name="config")

    # compose adapter lightning_module config.

    ## copy denoiser config from pretrained model to adapter config.
    diffusion_module_config = deepcopy(pretrained_config.lightning_module.diffusion_module)
    denoiser_config = diffusion_module_config.model

    with open_dict(adapter_config.adapter):
        for k, v in denoiser_config.items():
            # only legacy denoiser configs should contain property_embeddings_adapt
            if k != "_target_" and k != "property_embeddings_adapt":
                adapter_config.adapter[k] = v

            # do not adapt an existing <property_embeddings> field.
            if k == "property_embeddings":
                for field in v:
                    if field in adapter_config.adapter.property_embeddings_adapt:
                        adapter_config.adapter.property_embeddings_adapt.remove(field)

        # replace original GemNetT model with GemNetTCtrl model.
        adapter_config.adapter.gemnet["_target_"] = "mattergen.common.gemnet.gemnet_ctrl.GemNetTCtrl"

        # GemNetTCtrl model has additional input parameter condition_on_adapt, which needs to be set via property_embeddings_adapt.
        adapter_config.adapter.gemnet.condition_on_adapt = list(
            adapter_config.adapter.property_embeddings_adapt
        )

    # copy adapter config back into diffusion module config
    with open_dict(diffusion_module_config):
        diffusion_module_config.model = adapter_config.adapter
    with open_dict(lightning_module_config):
        lightning_module_config.diffusion_module = diffusion_module_config

    lightning_module = hydra.utils.instantiate(lightning_module_config)

    ckpt: dict = torch.load(ckpt_path, map_location=get_device())
    pretrained_dict: OrderedDict = ckpt["state_dict"]
    scratch_dict: OrderedDict = lightning_module.state_dict()
    scratch_dict.update(
        (k, pretrained_dict[k]) for k in scratch_dict.keys() & pretrained_dict.keys()
    )
    lightning_module.load_state_dict(scratch_dict, strict=True)

    # freeze pretrained weights if not full finetuning.
    if not adapter_config.full_finetuning:
        for name, param in lightning_module.named_parameters():
            if name in set(pretrained_dict.keys()):
                param.requires_grad_(False)

    return lightning_module, lightning_module_config


@hydra.main(
    config_path=str(MODELS_PROJECT_ROOT / "conf"), config_name="finetune", version_base="1.1"
)
def main(config: omegaconf.DictConfig):
    torch.set_float32_matmul_precision("high")
    # Make merged config options
    # CLI options take priority over YAML file options
    schema = OmegaConf.structured(Config)
    config = OmegaConf.merge(schema, config)
    OmegaConf.set_readonly(config, True)  # should not be written to
    print(OmegaConf.to_yaml(config, resolve=True))

    mattergen_finetune(config)


def mattergen_finetune(config: omegaconf.DictConfig):
    # Tensor Core acceleration (leads to ~2x speed-up during training)
    trainer: pl.Trainer = maybe_instantiate(config.trainer, pl.Trainer)
    datamodule: pl.LightningDataModule = maybe_instantiate(config.data_module, pl.LightningDataModule)

    # establish an adapter model
    pl_module, lightning_module_config = init_adapter_lightningmodule_from_pretrained(
        config.adapter, config.lightning_module
    )

    # replace denoiser config with adapter config.
    with open_dict(config):
        config.lightning_module = lightning_module_config

    config_as_dict = OmegaConf.to_container(config, resolve=True)
    print(json.dumps(config_as_dict, indent=4))
    # This callback will save a config.yaml file.
    trainer.callbacks.append(
        SaveConfigCallback(
            parser=SimpleParser(),
            config=config_as_dict,
            overwrite=True,
        )
    )
    # This callback will add a copy of the config to each checkpoint.
    trainer.callbacks.append(AddConfigCallback(config_as_dict))

    # Add support to compilation.
    if config.compile:
        torch.compile(pl_module, backend=config.compile_backend, mode=config.compile_mode)
    trainer.fit(
        model=pl_module,
        datamodule=datamodule,
        ckpt_path=None,
    )


if __name__ == "__main__":
    main()
