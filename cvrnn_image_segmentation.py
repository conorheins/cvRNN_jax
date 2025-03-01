import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.cluster import KMeans
jax.config.update("jax_enable_x64", True)

def gaussian_sheet(nrow, ncol, a, s, phi=None):
    """
    Create a weighted adjacency matrix based on a Gaussian function.
    
    Parameters:
      nrow, ncol: image dimensions.
      a: amplitude of the Gaussian.
      s: standard deviation.
      phi: (optional) phase offset (unused unless needed).
    
    Returns:
      W: a (nrow*ncol x nrow*ncol) weight matrix.
    """
    # Create evenly spaced positions from (1/nrow) to 1.
    drow = 1.0 / nrow
    dcol = 1.0 / ncol
    row = jnp.linspace(drow, 1.0, nrow)
    col = jnp.linspace(dcol, 1.0, ncol)
    # Use meshgrid with “ij” indexing so that the grid has shape (nrow,ncol)
    ROW, COL = jnp.meshgrid(row, col, indexing='ij')
    # Each pixel’s position as a two–column array (nrow*ncol, 2)
    pos = jnp.stack([ROW.flatten(), COL.flatten()], axis=-1)
    # Compute all pairwise Euclidean distances
    diff = pos[:, None, :] - pos[None, :, :]
    D = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))
    W = a * jnp.exp(-D**2 / (2 * s**2))
    if phi is not None:
        # (This extra branch reproduces the MATLAB “if nargin > 4” code.)
        W = jnp.where(W < 0.04, W * jnp.exp(1j * phi), W)
    return W


def run_2layer(im, a, s, nt, seed):
    """
    Run the two–layer CV‑RNN dynamics.
    
    Parameters:
      im: 2D image (Nr x Nc)
      a: tuple or list of amplitudes (layer1, layer2)
      s: tuple or list of standard deviations (layer1, layer2)
      nt: a two–element tuple/list (nt_layer1, nt_total)
      seed: integer seed for initialization
    
    Returns:
      save_x: array of shape (Nr*Nc, nt_total) holding the complex dynamics.
      mask: boolean array (Nr*Nc,) indicating background nodes.
    """
    Nr, Nc = im.shape
    # Build the layer 1 connection matrix.
    K1 = gaussian_sheet(Nr, Nc, a[0], s[0])
    
    # Set up the random key and generate the initial condition.
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    # Create a random phase uniformly in [–π, π]
    rand_angles = jax.random.uniform(subkey, shape=(Nr * Nc,), minval=-jnp.pi, maxval=jnp.pi)
    x0 = jnp.exp(1j * rand_angles)
    
    nt1, nt_total = nt[0], nt[1]
    
    # The intrinsic “frequency” for each node is taken from the input image.
    omega = im.flatten().astype(jnp.float64)
    
    # --- Layer 1 dynamics ---
    def step_fn(carry, _):
        x = carry
        x = (1j * omega).astype(jnp.complex64) * x + K1 @ x
        return x, x

    _, x_history = jax.lax.scan(step_fn, x0, None, length=nt1-1)
    x_history = jnp.concatenate((x0[None, :], x_history), axis=0)

    # Determine the background mask from the final phase of layer 1.
    phases = jnp.angle(x_history[-1])
    thr = jnp.mean(phases)
    count_above = jnp.sum(phases > thr)
    count_below = jnp.sum(phases < thr)
    # Use one mask rule or the other depending on which side is more populated.
    mask = jax.lax.cond(count_above > count_below,
                        lambda _: phases > thr,
                        lambda _: phases < thr,
                        operand=None)
    
    # --- Layer 2 dynamics ---
    # Build a new Gaussian sheet for layer 2.
    K2 = gaussian_sheet(Nr, Nc, a[1], s[1])
    # Zero out connections to/from nodes in the mask.
    valid = jnp.logical_not(mask)
    mask_float = valid.astype(jnp.float32)  # 1 for valid, 0 for masked.
    # Multiplying by the outer product zeros out rows/columns where a node is masked.
    K2 = K2 * jnp.outer(mask_float, mask_float)
    # Set omega to 0 for masked nodes.
    omega = jnp.where(mask, 0.0, omega)
    
    # Use the same initial condition for layer 2 but set masked nodes to 0.
    x02 = jnp.where(mask, 0.0, x0)
    def step_fn2(carry, _):
        x = carry
        x = (1j * omega) * x + K2 @ x
        return x, x

    _, x_history2 = jax.lax.scan(step_fn2, x02, None, length=nt_total-nt1)

    # For nodes in the mask, set the layer 2 dynamics to NaN.
    nan_complex = jnp.array(jnp.nan + 1j * jnp.nan, dtype=jnp.complex128)
    # Here we “broadcast” over time indices t >= nt1.
    time_idx = jnp.arange(nt_total)
    # Create a boolean mask for times in layer 2.
    time_mask = time_idx >= nt1

    total_history = jnp.concatenate((x_history, x_history2), axis=0).T
    # Expand the node mask to all timesteps and set those entries to NaN.
    total_history = jnp.where(jnp.outer(mask, time_mask), nan_complex, total_history)
    
    return total_history, mask


def spatiotemporal_segmentation(x, dims, win, ws, dw):
    """
    Use CV‑RNN dynamics to compute a similarity matrix and extract a low–dimensional projection.
    
    Parameters:
      x: array of shape (Nn, T) containing complex dynamics (for the valid nodes)
      dims: a list or tuple of dimensions (0–indexed) to extract (e.g. [0,1,2])
      win: tuple (start, end) of timesteps for analysis.
      ws: window size.
      dw: window step.
    
    Returns:
      rho: similarity matrices, shape (Nn, Nn, Nw)
      V: eigenvector matrices, shape (Nn, Nn, Nw)
      D: sorted eigenvalues, shape (Nn, Nw)
      prj: projection of the similarity matrix, shape (Nn, len(dims), Nw)
    """
    win_start, win_end = win[0]-1, win[1]
    window_starts = jnp.arange(win_start, win_end - ws + 1, dw)
    window_ends = window_starts + ws + 1
    Nw = window_starts.shape[0]
    
    rho_all = []
    V_all = []
    D_all = []
    prj_all = []
    
    for w in range(Nw):
        # Extract the time window for all nodes.
        x_window = x[:, window_starts[w]:window_ends[w]]
        # Compute the similarity matrix: each element is the (normalized) dot product over the window.
        rho_w = (x_window @ jnp.conjugate(x_window).T) / (ws+1)
        rho_all.append(rho_w)
        
        # Compute the eigen–decomposition.
        eigvals, eigvecs = jnp.linalg.eigh(rho_w)
        # Sort eigenvalues in descending order by absolute value.
        sorted_idx = jnp.argsort(jnp.abs(eigvals))[::-1]
        eigvals_sorted = eigvals[sorted_idx]
        eigvecs_sorted = eigvecs[:, sorted_idx]
        V_all.append(eigvecs_sorted)
        D_all.append(eigvals_sorted)
        
        # The projection is defined as: prj = real(rho) @ real(selected eigenvectors)
        prj_w = jnp.real(rho_w) @ jnp.real(eigvecs_sorted[:, dims])
        prj_all.append(prj_w)
    
    # Stack the results over the window dimension.
    rho_stack = jnp.stack(rho_all, axis=-1)
    V_stack = jnp.stack(V_all, axis=-1)
    D_stack = jnp.stack(D_all, axis=-1)
    prj_stack = jnp.stack(prj_all, axis=-1)
    
    return rho_stack, V_stack, D_stack, prj_stack


def plot_dynamics(x, layer1_final_time):
    """
    Animate the phase dynamics as an image.
    
    Parameters:
      x: a 3D array (Nr, Nc, T) holding phase values.
      layer1_final_time: time index used to adjust color limits.
    
    (Note: plotting is done with matplotlib and is not differentiable.)
    """
    Nr, Nc, T = x.shape
    plt.ion()  # turn interactive mode on
    fig, ax = plt.subplots()
    im_plot = ax.imshow(x[:, :, 0], cmap='hsv')
    ax.axis('off')
    t_text = ax.text(10, 10, '1 timesteps', color='white', fontsize=15)
    
    for t in range(1, T):
        im_plot.set_data(x[:, :, t])
        if t > layer1_final_time:
            im_plot.set_clim(-jnp.pi, jnp.pi)
        t_text.set_text(f'{t+1} timesteps')
        plt.pause(0.1)
    
    plt.ioff()
    plt.show()
    return ax

def main():
    # --- Set up hyperparameters ---
    alpha = (0.5, 0.5)
    sigma = (0.9, 0.0313)
    # In MATLAB: layer_1_time_range = 1:60 and layer_2_time_range = 61:200.
    # Here we simply pass the final time indices.
    layer_time_points = (60, 200)
    
    dims = [0, 1, 2]       # we extract three dimensions (Python indices 0,1,2)
    window_size = 40
    window_step = 40
    
    # --- Example 1: 2–shapes dataset ---
    # Load the dataset (assumes the .mat file contains variables 'images' and 'labels')
    data = loadmat('dataset/2shapes.mat')
    images = data['images']  # shape (Nr, Nc, num_images)
    labels = data['labels']
    
    # Choose one example (MATLAB uses 1-indexing; here we use index 0)
    image_number = 0
    im = images[:, :, image_number]
    lb = labels[:, :, image_number]
    
    # Run the cv-RNN dynamics.
    random_seed = 1
    # nt: (number of time steps for layer 1, total number of time steps)
    nt = (layer_time_points[0], layer_time_points[1])
    save_x, mask = run_2layer(im, alpha, sigma, nt, random_seed)
    
    # Reshape the saved dynamics to (Nr, Nc, T) and extract the phase.
    Nr, Nc = im.shape
    x_full = jnp.angle(save_x).reshape(Nr, Nc, -1)
    
    # --- Plot the input image ---
    plt.figure()
    plt.imshow(im, cmap='gray')
    plt.title('input')
    plt.axis('off')
    plt.show()
    
    # --- Animate the CV-RNN dynamics ---
    plot_dynamics(x_full, layer_time_points[0])
    
    # --- Spectral Clustering ---
    # Reshape the dynamics to (num_pixels, T) and “lift” them to the complex circle.
    x_reshaped = jnp.exp(1j * x_full.reshape(-1, x_full.shape[2]))
    # Consider only nodes that are not masked (i.e. not background)
    valid_indices = jnp.where(~mask.flatten())[0]
    x_valid = x_reshaped[valid_indices, :]
    
    # Run spatiotemporal segmentation on the valid nodes.
    win = (layer_time_points[0], layer_time_points[1])
    # x_valid = loadmat('/Users/conorheins/Documents/Verses/liboniEA2025image/x_valid.mat')['x']
    rho, V, D, prj = spatiotemporal_segmentation(x_valid, dims, win, window_size, window_step)
    
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

if __name__ == '__main__':
    main()
