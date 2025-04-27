# example_3shapes.py

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.cluster import KMeans
import argparse
import os

from cv_rnn import spatiotemporal_segmentation, plot_dynamics, gaussian_sheet
from cvrnn_layer import CVRNNLayer, MultiLayerCVRNN

def main(seed, visualize_dynamics=False):
    # Set matplotlib backend based on whether we're visualizing dynamics
    if not visualize_dynamics:
        matplotlib.use('Agg')
    else:
        matplotlib.use('TkAgg')  # or 'Qt5Agg' if you prefer

    # Create output directory if it doesn't exist
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # --- hyperparameters ---
    alpha = (0.5, 0.5)
    sigma = (0.9, 0.0313)
    nt1, nt2 = (60, 200)
    dims = [0, 1, 2]
    window_size = 40
    window_step = 40

    # --- load 3‑Shapes dataset ---
    data = loadmat('dataset/3shapes.mat')
    im = data['images'][:, :, 0]    # MATLAB image #1
    Nr, Nc = im.shape
    N = Nr * Nc
    # ground‑truth labels available in data['labels'] if you want to compute accuracy

     # --- Plot the input image ---
    plt.figure()
    plt.imshow(im, cmap='gray')
    plt.title('input')
    plt.axis('off')
    plt.savefig(f'{output_dir}/example_3shapes_input_seed_{seed}.png', bbox_inches='tight')
    plt.close()

    # build ω
    omega = im.flatten().astype(jnp.float64)

    # build B1, B2
    B1 = gaussian_sheet(Nr, Nc, alpha[0], sigma[0]).astype(jnp.complex128)
    B2 = gaussian_sheet(Nr, Nc, alpha[1], sigma[1]).astype(jnp.complex128)

    # instantiate layers & model
    layer1 = CVRNNLayer(B=B1, nt=nt1)
    layer2 = CVRNNLayer(B=B2, nt=nt2 - nt1)
    model  = MultiLayerCVRNN([layer1, layer2])

    # generate x0 using the same split pattern as run_2layer
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)

    # run multi-layer
    (h1, h2), (mask1, mask2) = model(omega=omega, key=subkey)

    # assemble full history and re-apply NaNs on masked nodes in layer 2
    combined = jnp.concatenate([h1, h2], axis=0)  # (nt2, N)
    nan_c = jnp.array(0+0j).at[()].set(complex(jnp.nan, jnp.nan))
    times = jnp.arange(nt2)
    tmask = times >= nt1              # shape (nt2,)
    mask_time = jnp.outer(tmask, mask1)  # (nt2, N)
    combined = jnp.where(mask_time, nan_c, combined)

    # transpose, angle, reshape
    X = combined.T                     # (N, nt2)
    x_full = jnp.angle(X).reshape(Nr, Nc, -1)

    # optional dynamics plot
    if visualize_dynamics:
        plot_dynamics(x_full, nt1)

    # --- spectral clustering ---
    # lift back onto complex unit circle
    x_flat = x_full.reshape(-1, x_full.shape[2])
    x_lift = jnp.exp(1j * x_flat)
    valid = ~mask1
    x_valid = x_lift[valid, :]

    rho, V, D, prj = spatiotemporal_segmentation(
        x_valid, dims, (nt1, nt2), window_size, window_step
    )

    # --- similarity projection (last window) ---
    w = prj.shape[-1] - 1
    colors = jnp.angle(x_valid[:, 120])  # e.g. 120th timestep
    prj_last = prj[:, :, w]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(
        np.array(prj_last[:, 0]),
        np.array(prj_last[:, 1]),
        np.array(prj_last[:, 2]),
        c=np.array(colors), cmap='hsv', s=50)
    ax.set_xlabel('dim1')
    ax.set_ylabel('dim2')
    ax.set_zlabel('dim3')
    plt.title('3‑Shapes similarity projection')
    fig.colorbar(sc, label='phase (rad)')
    plt.savefig(f'{output_dir}/example_3shapes_similarity_seed_{seed}.png', bbox_inches='tight')
    plt.close()

    # --- final k-means segmentation (3 clusters) ---
    predict = KMeans(n_clusters=3, random_state=seed).fit_predict(np.array(prj_last[:, :3]))
    
    # Build the segmented image.
    # Create an array of zeros for all pixels.
    segmented = np.zeros(N, dtype=np.float32)
    # Assign the predicted cluster labels to these valid indices, adding one to make sure each cluster gets its own color
    segmented[np.where(valid)[0]] = predict + 1
    segmented = segmented.reshape(Nr, Nc)

    plt.figure()
    # Here we "mask" the background (masked nodes remain 0) when displaying.
    plt.imshow(segmented, cmap='viridis')
    plt.title('3 shapes segmented')
    plt.axis('off')
    plt.savefig(f'{output_dir}/example_3shapes_segmented_seed_{seed}.png', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3-shapes segmentation example')
    parser.add_argument('--seed', type=int, default=5, help='random seed for run_2layer')
    parser.add_argument('--visualize_dynamics', action='store_true', help='whether to visualize the dynamics')
    args = parser.parse_args()
    main(args.seed, args.visualize_dynamics)