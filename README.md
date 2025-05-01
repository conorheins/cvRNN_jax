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

The codebase includes a common run script `run_cv_rnn.py` that can be used to run CVRNN-based segmentation on different datasets:

```bash
# Basic usage with default parameters
python run_cv_rnn.py --dataset 2shapes

# Run with custom seed
python run_cv_rnn.py --dataset 2shapes --seed 42

# Run with dynamics visualization
python run_cv_rnn.py --dataset 2shapes --visualize_dynamics

# Run with ensemble of 5 models evaluated on the same image in parallel
python run_cv_rnn.py --dataset 2shapes --ensemble_size 5

# Run specific image from dataset
python run_cv_rnn.py --dataset 2shapes --image_index 2

# Run with background pixels excluded for segmentation accuracy calculations
python run_cv_rnn.py --dataset 2shapes --exclude_background

# Combine multiple options
python run_cv_rnn.py --dataset 2shapes --ensemble_size 5 --image_index 2 --exclude_background
```

### Available Datasets

The script supports three different datasets, which are copied here for convenience from the [original MATLAB-based github repository](https://github.com/mullerlab/liboniEA2025image):

1. **2-Shapes Dataset** (`--dataset 2shapes`):
   - Demonstrates segmentation of images with two non-overlapping shapes
   - Includes three example images from the 2Shapes dataset

2. **3-Shapes Dataset** (`--dataset 3shapes`):
   - Demonstrates segmentation of images with three distinct shapes
   - Includes three example images from the 3Shapes dataset

3. **Natural Image Dataset** (`--dataset natural_image`):
   - Demonstrates segmentation of a natural image
   - Note: `--image_index` is not available for this dataset

### Command Line Arguments

The unified script accepts the following arguments:

- `--dataset`: Required. Choose from ["2shapes", "3shapes", "natural_image"]
- `--seed INT`: Specify a random seed for reproducibility (default: 1)
- `--visualize_dynamics`: Enable visualization of the CV-RNN dynamics (interactive plot)
- `--ensemble_size INT`: Number of models to run in parallel (default: 1)
- `--exclude_background`: Exclude background pixels when computing accuracy metrics
- `--image_index INT`: Index of the image to use from the dataset (default: 0, not available for natural image)

## Outputs

The script produces several output files in the `output/{dataset}_img{image_index}_seed{seed}/` directory:

1. Input image visualization (`input.png`)
2. 3D similarity projection plot (`sim_{i}.png` for each model in ensemble)
3. Final segmentation result (`seg_{i}.png` for each model in ensemble)

When using `--ensemble_size > 1`:
- Separate output files for each model in the ensemble
- Mean and standard deviation of segmentation accuracy across the ensemble

When using the `--visualize_dynamics` flag, an interactive animation showing the phase dynamics will be displayed.

