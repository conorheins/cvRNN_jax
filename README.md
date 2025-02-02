## cvRNN_jax 

This is a direct JAX re-implementation of some of the MATLAB code accompaying the paper [Liboni*, Budzinski*, Busch*, Löwe, Keller, Welling, and Muller (2025) Image segmentation with traveling waves in an exactly solvable recurrent neural network. PNAS 122: e2321319121 *equal contribution](https://www.pnas.org/doi/10.1073/pnas.2321319121)


The original MATLAB source code can be found at the following repository: https://github.com/mullerlab/liboniEA2025image

## Usage

Run the main script using 
```
>> python cvrnn_image_segmentation.py
```

So far only the 2-shapes dataset example is run.

Requirements are `jax`, `jaxlib`, `numpy`, `matplotlib`, `scipy` and `sklearn`. 
