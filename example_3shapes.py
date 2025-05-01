# example_3shapes.py
import jax 
jax.config.update("jax_enable_x64", True)

import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat
from jax import numpy as jnp
from jax import vmap
import equinox as eqx

from cvrnn_layer import CVRNNLayer, MultiLayerCVRNN
from cv_rnn import spatiotemporal_segmentation, plot_dynamics, gaussian_sheet
from sklearn.cluster import KMeans
import argparse
import os

def main(seed, visualize_dynamics=False, ensemble_size=1):
    # Set matplotlib backend based on whether we're visualizing dynamics
    if not visualize_dynamics:
        matplotlib.use('Agg')
    else:
        matplotlib.use('TkAgg')  # or 'Qt5Agg' if you prefer

    # Create output directory if it doesn't exist
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # --- hyperparameters ---
    alpha = (0.5 * jnp.ones(ensemble_size), 0.5 * jnp.ones(ensemble_size))
    sigma = (0.9 * jnp.ones(ensemble_size), 0.0313 * jnp.ones(ensemble_size))
    nt1, nt2 = (60, 200)
    dims = [0, 1, 2]
    window_size = 40
    window_step = 40

    # --- load 3‑Shapes dataset ---
    data = loadmat('dataset/3shapes.mat')
    im = data['images'][:, :, 0]    # MATLAB image #1
    labels_gt = data['labels'][:,:,0].astype(np.int32)
    labels_flat = labels_gt.flatten()
    Nr, Nc = im.shape
    N = Nr * Nc

    seed_specific_dir = os.path.join(output_dir, f'example_3shapes_seed_{seed}')
    os.makedirs(seed_specific_dir, exist_ok=True)

    # --- Plot the input image ---
    plt.figure()
    plt.imshow(im, cmap='gray')
    plt.title('input')
    plt.axis('off')
    plt.savefig(os.path.join(seed_specific_dir, 'input.png'), bbox_inches='tight')
    plt.close()

    # build ω
    omega = im.flatten().astype(jnp.float64)

    # build B1, B2 (vmap initialization across alpha and sigma parameters over ensemble dimension)
    B1s = vmap(gaussian_sheet, in_axes=(None, None, 0, 0))(Nr, Nc, alpha[0], sigma[0]).astype(jnp.complex128)
    B2s = vmap(gaussian_sheet, in_axes=(None, None, 0, 0))(Nr, Nc, alpha[1], sigma[1]).astype(jnp.complex128)

    # manufacture an ensemble of models, each with its own B and x0
    @eqx.filter_vmap
    def make_model(key_i, B1_i, B2_i):
        l1 = CVRNNLayer(B=B1_i, nt=nt1, key=key_i)
        l2 = CVRNNLayer(B=B2_i, nt=nt2-nt1)
        return MultiLayerCVRNN([l1, l2])

    # prepare PRNGKeys
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, ensemble_size)

    models = make_model(keys, B1s, B2s)

    # run the entire ensemble on the same omega
    @eqx.filter_vmap(in_axes=(eqx.if_array(0), None))
    def run_ensemble(model, omega):
        # returns ( (h1,h2), (mask1,mask2) )
        return model(omega=omega)

    (h1s, h2s), (mask1s, mask2s) = run_ensemble(models, omega)

    # do the rest of the plotting / etc. for each model's output of the ensemble
    accuracies = []
    for ii in range(ensemble_size):
        # assemble full history and re-apply NaNs on masked nodes in layer 2
        combined = jnp.concatenate([h1s[ii], h2s[ii]], axis=0)  # (nt2, N)
        nan_c = jnp.array(0+0j).at[()].set(complex(jnp.nan, jnp.nan))
        times = jnp.arange(nt2)
        tmask = times >= nt1              # shape (nt2,)
        mask_time = jnp.outer(tmask, mask1s[ii])  # (nt2, N)
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
        valid = ~mask1s[ii]
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
        plt.savefig(os.path.join(seed_specific_dir, f'similarity_model_{ii}.png'), bbox_inches='tight')
        plt.close()

        # --- final k-means segmentation (3 clusters) ---
        predict = KMeans(n_clusters=3, random_state=seed).fit_predict(np.array(prj_last[:, :3]))
        
        # Build the segmented image.
        # Create an array of zeros for all pixels.
        segmented = np.zeros(N, dtype=np.int32)
        # Assign the predicted cluster labels to these valid indices, adding one to make sure each cluster gets its own color
        segmented[np.where(valid)[0]] = predict + 1
        segmented = segmented.reshape(Nr, Nc)

        plt.figure()
        # Here we "mask" the background (masked nodes remain 0) when displaying.
        plt.imshow(segmented, cmap='viridis')
        plt.title('3 shapes segmented')
        plt.axis('off')
        plt.savefig(os.path.join(seed_specific_dir, f'segmented_model_{ii}.png'), bbox_inches='tight')
        plt.close()

        # --- compute pixel-wise accuracy via Hungarian matching ---
        y_pred = segmented.flatten().astype(np.int32)
        y_true = labels_flat.copy()
        # restrict to non-background if requested
        if args.exclude_background:
            valid = (labels_flat!=0).astype(bool)  # True = valid
            idxs = np.where(valid)[0]
            y_pred = y_pred[idxs]
            y_true = y_true[idxs]

        # find unique labels in pred/true
        pred_labels = np.unique(y_pred)
        true_labels = np.unique(y_true)
        # build the confusion matrix
        M = np.zeros((pred_labels.size, true_labels.size), dtype=np.int32)
        for i, p in enumerate(pred_labels):
            for j, t in enumerate(true_labels):
                M[i, j] = np.sum((y_pred == p) & (y_true == t))

        # Hungarian to maximize trace(M)
        row_ind, col_ind = linear_sum_assignment(M.max() - M)
        total = M[row_ind, col_ind].sum()
        acc = total / y_true.size
        accuracies.append(acc)
    
    accuracies = jnp.array(accuracies)
    mean_acc = float(accuracies.mean())
    var_acc  = float(accuracies.std())
    print(f"\nOverall accuracy: mean={mean_acc:.4f}, standard deviation={var_acc:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3-shapes segmentation example')
    parser.add_argument('--seed', type=int, default=5, help='random seed for run_2layer')
    parser.add_argument('--visualize_dynamics', action='store_true', help='whether to visualize the dynamics')
    parser.add_argument('--exclude_background', action='store_true', help='only compute accuracy on non-background pixels')
    parser.add_argument('--ensemble_size', type=int, default=1, help='size of the model ensemble (number of models to run in parallel with vmap)')
    args = parser.parse_args()
    main(args.seed, args.visualize_dynamics, args.ensemble_size)