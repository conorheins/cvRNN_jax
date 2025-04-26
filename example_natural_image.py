# example_natural_image.py

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

from cv_rnn import run_2layer, spatiotemporal_segmentation, plot_dynamics

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
    layer_time_points = (60, 200)
    dims = [0, 1, 2]
    window_size = 40
    window_step = 40

    # --- load natural image dataset ---
    data = loadmat('dataset/natural_image.mat')
    # Original image and preprocessed input image
    original_image = data['in']  # Original image
    im = data['im']              # Preprocessed input for cv-RNN

    # --- Plot the original image ---
    plt.figure()
    plt.imshow(original_image, cmap='gray')
    plt.title('original image')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f'example_natural_image_original_seed_{seed}.png'), bbox_inches='tight')
    plt.close()

    # --- Plot the input image ---
    plt.figure()
    plt.imshow(im, cmap='gray')
    plt.title('input')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f'example_natural_image_input_seed_{seed}.png'), bbox_inches='tight')
    plt.close()

    # --- run cv-RNN ---
    nt = layer_time_points
    X, mask = run_2layer(im, alpha, sigma, nt, seed=seed)

    # reshape and extract phase
    Nr, Nc = im.shape
    x_full = jnp.angle(X).reshape(Nr, Nc, -1)

    # visualize dynamics if requested
    if visualize_dynamics:
        plot_dynamics(x_full, layer_time_points[0])

    # --- spectral clustering ---
    # lift back onto complex unit circle
    x_flat = x_full.reshape(-1, x_full.shape[2])
    x_lift = jnp.exp(1j * x_flat)
    valid_indices = jnp.where(~mask.flatten())[0]
    x_valid = x_lift[valid_indices, :]

    rho, V, D, prj = spatiotemporal_segmentation(
        x_valid, dims, layer_time_points, window_size, window_step
    )

    # --- similarity projection ---
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
    ax.set_xlabel('dimension 1')
    ax.set_ylabel('dimension 2')
    ax.set_zlabel('dimension 3')
    plt.title('similarity projection')
    fig.colorbar(sc, label='phase (rad)')
    plt.savefig(os.path.join(output_dir, f'example_natural_image_similarity_seed_{seed}.png'), bbox_inches='tight')
    plt.close()

    # --- final segmentation via k‑means (2 clusters) ---
    predict = KMeans(n_clusters=2).fit_predict(np.array(prj_last[:, :3]))
    
    # Build the segmented image.
    # Create an array of zeros for all pixels.
    segmented_image = np.zeros(mask.flatten().shape, dtype=np.float32)
    # Assign the predicted cluster labels to these valid indices, adding one to make sure each cluster gets its own color
    segmented_image[valid_indices] = predict+1
    segmented_image = segmented_image.reshape(Nr, Nc)

    plt.figure()
    # Here we "mask" the background (masked nodes remain 0) when displaying.
    plt.imshow(segmented_image, cmap='viridis')
    plt.title('segmented image')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f'example_natural_image_segmented_seed_{seed}.png'), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Natural image segmentation example')
    parser.add_argument('--seed', type=int, default=1, help='random seed for run_2layer')
    parser.add_argument('--visualize_dynamics', action='store_true', help='whether to visualize the dynamics')
    args = parser.parse_args()
    main(args.seed, args.visualize_dynamics) 