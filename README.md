# cvRNN_jax 

This is a direct JAX re-implementation of the MATLAB code accompaying the paper [Liboni*, Budzinski*, Busch*, Löwe, Keller, Welling, and Muller (2025) Image segmentation with traveling waves in an exactly solvable recurrent neural network. PNAS 122: e2321319121 *equal contribution](https://www.pnas.org/doi/10.1073/pnas.2321319121)

The original MATLAB source code can be found at the following repository: https://github.com/mullerlab/liboniEA2025image

## Implementation

This repository provides two implementation approaches:
1. A functional JAX implementation in `cv_rnn.py`
2. An object-oriented implementation using Equinox in `cvrnn_layer.py`, featuring:
   - `CVRNNLayer`: A single-layer implementation with various initialization options
   - `MultiLayerCVRNN`: A multi-layer implementation that supports layer stacking

## Requirements

- jax
- jaxlib
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
```

## Command Line Arguments

Each example script accepts the following arguments:

- `--seed INT`: Specify a random seed for reproducibility (default values vary by example)
- `--visualize_dynamics`: Enable visualization of the CV-RNN dynamics (interactive plot)

## Outputs

Each script produces several output files in the `output/` directory:

1. Input image visualization
2. 3D similarity projection plot
3. Final segmentation result

When using the `--visualize_dynamics` flag, an interactive animation showing the phase dynamics will be displayed.

