# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Generic, Mapping, Tuple, TypeVar

import torch

from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.sampling.pc_sampler import (
    PredictorCorrector, SampleAndMean, SampleAndMeanAndRecords,
    _sample_prior, _mask_replace
)
from mattergen.diffusion.corruption.multi_corruption import apply

Diffusable = TypeVar("Diffusable", bound=BatchedData)


class InpaintingPredictorCorrector(PredictorCorrector[Diffusable], Generic[Diffusable]):
    """
    Inpainting sampler based on the RePaint paper technique.
    
    This class extends the standard PredictorCorrector sampler to support inpainting
    where part of the structure is known (e.g., bulk section) and part is unknown 
    (e.g., surface or interface section).
    """

    def __init__(
        self,
        *,
        n_resample_iters: int = 1,
        **kwargs,
    ):
        """
        Initialize the inpainting sampler with resampling iterations.
        
        Args:
            n_resample_iters: Number of resampling iterations per denoising step (RePaint technique)
            **kwargs: Additional arguments passed to the parent PredictorCorrector class
        """
        super().__init__(**kwargs)
        self.n_resample_iters = n_resample_iters

    def sample(
        self, 
        conditioning_data: BatchedData, 
        reference_data: BatchedData,
        mask: Mapping[str, torch.Tensor] | None = None,
    ) -> SampleAndMean:
        """
        Create samples using inpainting technique based on RePaint paper.
        
        Args:
            conditioning_data: Batched conditioning data
            reference_data: Reference data for known regions
            mask: Float mask between 0 and 1,
                where 1 indicates known (fixed) regions and 0 indicates unknown regions.
                        
        Returns:
           (batch, mean_batch). The difference between these is that `mean_batch` has no noise added at the final denoising step.
        """
        return self._sample_maybe_record_with_inpainting(
            conditioning_data, reference_data, mask=mask, record=False
        )[:2]

    def sample_with_record(
        self, 
        conditioning_data: BatchedData,
        reference_data: BatchedData,
        mask: Mapping[str, torch.Tensor] | None = None,
    ) -> SampleAndMeanAndRecords:
        """
        Create samples using inpainting technique based on RePaint paper and record intermediate steps.
        
        Args:
            conditioning_data: Batched conditioning data
            reference_data: Reference data for known regions
            mask: Float mask between 0 and 1,
                where 1 indicates known (fixed) regions and 0 indicates unknown regions.
                        
        Returns:
           (batch, mean_batch, recorded_samples).
        """
        return self._sample_maybe_record_with_inpainting(
            conditioning_data, reference_data, mask=mask, record=True
        )

    def _sample_maybe_record_with_inpainting(
        self,
        conditioning_data: BatchedData,
        reference_data: BatchedData,
        mask: Mapping[str, torch.Tensor] | None = None,
        record: bool = False,
    ) -> SampleAndMeanAndRecords:
        """
        Create samples using inpainting technique based on RePaint paper.
        
        Args:
            conditioning_data: Batched conditioning data
            reference_data: Reference data for known regions
            mask: Float mask between 0 and 1,
                where 1 indicates known (fixed) regions and 0 indicates unknown regions.
            record: Whether to record intermediate samples
                        
        Returns:
           (batch, mean_batch, recorded_samples).
        """
        if isinstance(self._diffusion_module, torch.nn.Module):
            self._diffusion_module.eval()
        mask = mask or {}
        conditioning_data = conditioning_data.to(self._device)
        reference_data = reference_data.to(self._device)
        mask = {k: v.to(self._device) for k, v in mask.items()}
        # Use regular prior sampling. No special inpainting initialization needed.
        batch = _sample_prior(self._multi_corruption, conditioning_data, mask=mask)
        return self._denoise_with_inpainting(
            batch=batch, 
            mask=mask, 
            reference_data=reference_data,
            record=record
        )

    def _denoise_with_inpainting(
        self,
        batch: Diffusable,
        mask: dict[str, torch.Tensor],
        reference_data: Diffusable,
        record: bool = False,
    ) -> SampleAndMeanAndRecords:
        """
        Denoise with inpainting technique based on RePaint paper.
        In each denoising step, the known parts are noised from reference data under corruption schedule,
        added with the unknown segment, then the whole image undergoes one step denoise using the score network,
        but only the known part under mask is passed to the next denoising step while the unknown part 
        will still use scheduled corruption from its original data.
        
        Args:
            batch: The initial noisy batch
            mask: Float mask between 0 and 1,
                where 1 indicates known (fixed) regions and 0 indicates unknown regions.
            reference_data: The original reference data for known regions
            record: Whether to record intermediate samples
        """
        recorded_samples = None
        if record:
            recorded_samples = []
        for k in self._predictors:
            mask.setdefault(k, None)
        for k in self._correctors:
            mask.setdefault(k, None)
        mean_batch = batch.clone()
        
        # Initialize mask weights.
        mask = {k: msk.float() if msk is not None else None for k, msk in mask.items()}

        # Decreasing timesteps from T to eps_t
        timesteps = torch.linspace(self._max_t, self._eps_t, self.N, device=self._device)
        dt = -torch.tensor((self._max_t - self._eps_t) / (self.N - 1)).to(self._device)

        for i in range(self.N):
            # Set the timestep
            t = torch.full((batch.get_batch_size(),), timesteps[i], device=self._device)
            
            # For inpainting, we need to noise the known parts of reference data at current timestep
            # and combine with the unknown parts from current batch
            if i > 0:  # Not needed for the first step since batch is already properly initialized
                # Noise the reference data at current timestep
                noised_reference = self._multi_corruption.sample_marginal(reference_data, t)
                
                # Combine noised reference data (for known parts) with current batch (for unknown parts)
                combined_batch = batch.clone()
                for k in self._multi_corruption.corrupted_fields:
                    if mask.get(k) is not None:
                        # Use weight-based interpolation
                        # Weighted combination based on distance to interface
                        combined_batch[k] = (
                            mask[k] * noised_reference[k] +
                            (1 - mask[k]) * batch[k]
                        )
                batch = combined_batch

            # RePaint resampling iterations
            for resample_iter in range(self.n_resample_iters):
                # Corrector updates.
                if self._correctors:
                    for _ in range(self._n_steps_corrector):
                        score = self._score_fn(batch, t)
                        fns = {
                            k: corrector.step_given_score for k, corrector in self._correctors.items()
                        }
                        samples_means: dict[str, Tuple[torch.Tensor, torch.Tensor]] = apply(
                            fns=fns,
                            broadcast={"t": t, "dt": dt},
                            x=batch,
                            score=score,
                            batch_idx=self._multi_corruption._get_batch_indices(batch),
                        )
                        if record and resample_iter == self.n_resample_iters - 1:  # Only record on last iteration
                            recorded_samples.append(batch.clone().to("cpu"))
                        batch, mean_batch = _mask_replace(
                            samples_means=samples_means, 
                            batch=batch, 
                            mean_batch=mean_batch, 
                            mask=mask,
                        )

                # Predictor updates
                score = self._score_fn(batch, t)
                predictor_fns = {
                    k: predictor.update_given_score for k, predictor in self._predictors.items()
                }
                samples_means = apply(
                    fns=predictor_fns,
                    x=batch,
                    score=score,
                    broadcast=dict(t=t, batch=batch, dt=dt),
                    batch_idx=self._multi_corruption._get_batch_indices(batch),
                )
                
                if record and resample_iter == self.n_resample_iters - 1:  # Only record on last iteration
                    recorded_samples.append(batch.clone().to("cpu"))
                batch, mean_batch = _mask_replace(
                    samples_means=samples_means, 
                    batch=batch, 
                    mean_batch=mean_batch, 
                    mask=mask,
                )
                # For all but the last resample iteration, add noise to revert one step back to time t
                if resample_iter < self.n_resample_iters - 1:
                    # Add noise to revert the prediction back to time t for another denoising iteration
                    batch = self._multi_corruption.sample_next_timestep(batch, t-dt, dt)

        return batch, mean_batch, recorded_samples
