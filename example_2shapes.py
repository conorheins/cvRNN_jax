# example_2shapes.py
import jax 
jax.config.update("jax_enable_x64", True)

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from jax import numpy as jnp
from cv_rnn import run_2layer, spatiotemporal_segmentation, plot_dynamics
from sklearn.cluster import KMeans

def main():
    # hyperparams (same for both examples)
    alpha = (0.5, 0.5)
    sigma = (0.9, 0.0313)
    layer_time_points = (60, 200)
    dims = [0,1,2]
    window_size, window_step = 40, 40

    # 2‑Shapes
    data = loadmat('dataset/2shapes.mat')
    im = data['images'][:, :, 0]        # MATLAB’s image 1
    nt = layer_time_points
    X, mask = run_2layer(im, alpha, sigma, nt, seed=1)

    # reshape & phase
    Nr, Nc = im.shape
    x_full = jnp.angle(X).reshape(Nr, Nc, -1)

    # --- Plot the input image ---
    plt.figure()
    plt.imshow(im, cmap='gray')
    plt.title('input')
    plt.axis('off')
    plt.show()

    plot_dynamics(x_full, layer_time_points[0])

    # spectral clustering
    x = jnp.exp(1j * x_full.reshape(-1, x_full.shape[2]))
    valid_indices = jnp.where(~mask.flatten())[0]
    x_valid = x[valid_indices, :]
    rho, V, D, prj = spatiotemporal_segmentation(x_valid, dims, layer_time_points, window_size, window_step)

    # --- Visualize the similarity projection ---
    # (Here we select the last time window.)
    w = prj.shape[-1] - 1
    colors = jnp.angle(x_valid[:, 120])  # use a time index (e.g. 120th timestep)
    prj_last = prj[:, :, w]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(np.array(prj_last[:, 0]),
                    np.array(prj_last[:, 1]),
                    np.array(prj_last[:, 2]),
                    c=np.array(colors), cmap='hsv', s=50)
    ax.set_xlabel('dimension 1')
    ax.set_ylabel('dimension 2')
    ax.set_zlabel('dimension 3')
    plt.title('similarity projection')
    fig.colorbar(sc, label='phase (rad)')
    plt.show()
    
    # --- Final segmentation using k-means ---
    kmeans = KMeans(n_clusters=2)
    predict = kmeans.fit_predict(np.array(prj_last[:, :3]))
    
    # Build the segmented image.
    # Create an array of zeros for all pixels.
    segmented_image = np.zeros(mask.flatten().shape, dtype=np.float32)

    # Assign the predicted cluster labels to these valid indices, adding one to make sure each cluster gets its own color
    segmented_image[valid_indices] = predict+1

    # Reshape the segmented_image to the original image dimensions.
    segmented_image = segmented_image.reshape(Nr, Nc)

    plt.figure()
    # Here we “mask” the background (masked nodes remain 0) when displaying.
    plt.imshow(segmented_image, cmap='viridis')
    plt.title('segmented image')
    plt.axis('off')
    plt.show()

if __name__=='__main__':
    main()