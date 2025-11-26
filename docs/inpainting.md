# Crystal Structure Inpainting with MatterGen

This document explains how to use the inpainting functionality implemented for MatterGen based on the RePaint technique.

## Overview

The inpainting technique allows you to generate physically plausible surface and interfacial atomic structures by:
1. Separating a crystal structure into known (bulk) and unknown (surface/interface) regions using a mask
2. In each denoising step:
   - Noising the known parts from reference data under the corruption schedule
   - Combining with the unknown parts 
   - Applying one step of denoising using the score network
   - Only passing the known parts under mask to the next step while keeping unknown parts corrupted
3. Using a non-binary weight mask for smooth transitions based on distance to the interface

## Implementation Details

### Core Classes

1. **InpaintingPredictorCorrector** - Main class for inpainting sampling
2. **Extensions to PredictorCorrector** - Added methods to the base class for inpainting functionality

### Key Methods

- `sample_with_inpainting()` - Generate samples using inpainting
- `sample_with_inpainting_and_record()` - Generate samples and record intermediate steps
- `_denoise_with_inpainting()` - Core denoising algorithm with inpainting
- `_mask_replace_weighted()` - Apply weighted masks for smooth transitions
- `_add_noise_to_revert_to_time_t()` - Add noise to revert prediction back to time t (RePaint resampling)

## Usage

### 1. Prepare Your Data

```python
# Load or create your reference crystal structure
reference_structure = load_your_crystal_structure()

# Create masks for known and unknown regions
masks, weight_masks = prepare_inpainting_masks(reference_structure, interface_plane_z)
```

### 2. Create the Inpainting Sampler

```python
from mattergen.diffusion.sampling.inpainting_sampler import InpaintingPredictorCorrector

sampler = InpaintingPredictorCorrector.from_pl_module(
    pl_module=your_trained_model,
    N=1000,  # Number of denoising steps
    eps_t=1e-3,
    n_resample_iters=10,  # Number of resampling iterations per denoising step (RePaint technique)
    # Add predictor and corrector configurations
)
```

### 3. Generate Structures

```python
# Generate with inpainting
generated_batch, mean_batch = sampler.sample_with_inpainting(
    conditioning_data=reference_batch,
    reference_data=reference_batch,
    mask=masks,
    weight_mask=weight_masks
)
```

## RePaint Resampling Technique

The RePaint resampling technique improves the quality of inpainted regions by performing multiple denoising iterations at each timestep. At each timestep t:

1. Predict the structure at t-1 using the score network
2. Add noise to revert the prediction back to time t
3. Repeat the denoising process for a specified number of iterations
4. Use the final prediction for the next timestep

This technique helps blend the unknown parts of the structure more naturally with the known parts.

## Mask Preparation

### Binary Masks
- 1: Known (fixed) regions
- 0: Unknown (to be generated) regions

### Weight Masks
- Continuous values between 0 and 1
- 1: Known regions
- Decaying toward 0: Far from interface for smooth transitions

Example weight mask creation:
```python
def create_distance_weight_mask(positions, interface_center, decay_length=2.0):
    distances = torch.norm(positions - interface_center, dim=1)
    weights = torch.exp(-distances / decay_length)  # Exponential decay
    return weights
```

## Technical Details

The inpainting algorithm works as follows:

1. **Initialization**: Start with noise combined with reference data according to masks
2. **Denoising Loop**: For each timestep:
   a. Noise the known parts of reference data at current timestep
   b. Combine with unknown parts from current batch
   c. Apply RePaint resampling iterations:
      i. Apply predictor-corrector denoising step
      ii. Add noise to revert prediction back to time t
      iii. Repeat for n_resample_iters
   d. Apply weighted mask to preserve known information with smooth transitions
3. **Output**: Fully denoised structure with known regions preserved and unknown regions generated

## Advanced Usage

### Custom Weight Functions

You can implement custom weight functions for specific interface geometries:
```python
def custom_weight_function(positions, interface_params):
    # Implement your specific weight calculation
    # based on interface geometry
    pass
```

### Multi-Region Inpainting

For complex structures with multiple interface regions:
```python
# Create separate masks for each region
bulk_mask = create_mask_for_bulk_region()
surface_mask = create_mask_for_surface_region()
interface_mask = create_mask_for_interface_region()

# Combine as needed
```

## Limitations and Considerations

1. **Memory Usage**: Recording intermediate steps can consume significant memory
2. **Computational Cost**: Inpainting requires additional computations compared to standard sampling
3. **Mask Design**: The quality of results depends heavily on proper mask design
4. **Weight Functions**: Smooth transition quality depends on appropriate weight function choice
5. **Resampling Iterations**: More resampling iterations improve quality but increase computation time

## Troubleshooting

### Common Issues

1. **Poor Interface Quality**: Check weight mask parameters and decay functions
2. **Discontinuities**: Ensure smooth weight transitions and sufficient denoising steps
3. **Memory Issues**: Reduce batch size or disable recording of intermediate steps
4. **Slow Performance**: Reduce n_resample_iters if computation time is too high

### Debugging Tips

1. Visualize masks to ensure correct regions are identified
2. Record intermediate steps to observe denoising progression
3. Experiment with different weight functions and parameters
4. Try different values of n_resample_iters (1-20) to balance quality and performance