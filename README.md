# cvRNN_jax 

This is a JAX implementation of the complex-valued RNN model used unsupervised object segmentation, originally based on the MATLAB implementation accompanying the 2024 publication in PNAS:

[Liboni*, Budzinski*, Busch*, Löwe, Keller, Welling, and Muller (2025) Image segmentation with traveling waves in an exactly solvable recurrent neural network. PNAS 122: e2321319121 *equal contribution](https://www.pnas.org/doi/10.1073/pnas.2321319121)

The original MATLAB source code can be found at the following repository: https://github.com/mullerlab/liboniEA2025image

## Implementation

This repository provides two implementation approaches:
1. A functional JAX implementation in `cv_rnn.py`
2. An object-oriented implementation using Equinox in `cvrnn_layer.py`, featuring:
   - `CVRNNLayer`: A single-layer implementation with various initialization options
   - `MultiLayerCVRNN`: A multi-layer implementation that supports layer stacking via composition of multiple instances of single `CVRNNLayer` instances.

## Requirements

- jax
- jaxlib
- jaxtyping
- equinox
- numpy
- matplotlib
- scipy
- scikit-learn (sklearn)

## Usage

The codebase includes three example scripts that demonstrate the CV-RNN approach on different datasets:

### 2-Shapes Example

Demonstrates segmentation of images with two distinct shapes:

```bash
# Run with default seed
python example_2shapes.py

# Run with custom seed
python example_2shapes.py --seed 42

# Run with dynamics visualization
python example_2shapes.py --visualize_dynamics

# Run with ensemble of 5 models evaluated on the same image in parallel
python example_2shapes.py --ensemble_size 5

# Run specific image from dataset
python example_2shapes.py --image_index 2

# Run with background pixels excluded from accuracy
python example_2shapes.py --exclude_background

# Combine multiple options
python example_2shapes.py --ensemble_size 5 --image_index 2 --exclude_background
```

### 3-Shapes Example

Demonstrates segmentation of images with three distinct shapes:

```bash
# Run with default seed
python example_3shapes.py

# Run with custom seed
python example_3shapes.py --seed 42

# Run with dynamics visualization
python example_3shapes.py --visualize_dynamics

# Run with ensemble of 5 models
python example_3shapes.py --ensemble_size 5

# Run specific image from dataset
python example_3shapes.py --image_index 2

# Run with background pixels excluded from accuracy
python example_3shapes.py --exclude_background

# Combine multiple options
python example_3shapes.py --ensemble_size 5 --image_index 2 --exclude_background
```

### Natural Image Example

Demonstrates segmentation of a natural image:

```bash
# Run with default seed
python example_natural_image.py

# Run with custom seed
python example_natural_image.py --seed 42

# Run with dynamics visualization
python example_natural_image.py --visualize_dynamics

# Run with ensemble of 5 models
python example_natural_image.py --ensemble_size 5

# Run with background pixels excluded from accuracy
python example_natural_image.py --exclude_background

# Combine multiple options
python example_natural_image.py --ensemble_size 5 --exclude_background
```

## Command Line Arguments

Each example script accepts the following arguments:

- `--seed INT`: Specify a random seed for reproducibility (default values vary by example)
- `--visualize_dynamics`: Enable visualization of the CV-RNN dynamics (interactive plot)
- `--ensemble_size INT`: Number of models to run in parallel (default: 1)
- `--exclude_background`: Exclude background pixels when computing accuracy metrics
- `--image_index INT`: Index of the image to use from the dataset (default: 0, not available for natural image example)

## Outputs

Each script produces several output files in the `output/` directory:

1. Input image visualization
2. 3D similarity projection plot
3. Final segmentation result
4. When using `--ensemble_size > 1`:
   - Separate output files for each model in the ensemble
   - Mean and standard deviation of segmentation accuracy across the ensemble

When using the `--visualize_dynamics` flag, an interactive animation showing the phase dynamics will be displayed.

