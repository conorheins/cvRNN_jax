# run_cv_rnn.py

import os
import argparse

import jax            
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from jax import vmap
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.pyplot as plt

from cv_rnn import spatiotemporal_segmentation, plot_dynamics, gaussian_sheet
from cvrnn_layer import CVRNNLayer, MultiLayerCVRNN

def load_dataset(name, idx):
    data = loadmat(f"dataset/{name}.mat")
    if name in ("2shapes","3shapes"):
        im     = data["images"][:,:,idx]
        labels = data["labels"][:,:,idx].astype(np.int32)
        # number of clusters = unique non-0 labels
        n_clusters = int(np.unique(labels[labels!=0]).size)
        original = None
    elif name=="natural_image":
        im       = data["im"]
        original = data["in"]
        labels   = data["lb"].astype(np.int32)
        n_clusters = 2
    else:
        raise ValueError(f"Unknown dataset {name}")
    return im, original, labels, n_clusters

def main(
    *,
    dataset: str,
    image_index: int,
    seed: int,
    ensemble_size: int,
    exclude_background: bool,
    visualize_dynamics: bool,
):
    # backend selection
    if not visualize_dynamics:
        matplotlib.use("Agg")
    else:
        matplotlib.use("TkAgg")

    # hyper‐parameters (same for all datasets)
    alpha = (0.5 * jnp.ones(ensemble_size), 0.5 * jnp.ones(ensemble_size))
    sigma = (0.9 * jnp.ones(ensemble_size), 0.0313 * jnp.ones(ensemble_size))
    nt1, nt2 = 60, 200
    dims = [0,1,2]
    window_size, window_step = 40, 40

    # load
    im, original_image, labels_gt, n_clusters = load_dataset(dataset, image_index)
    Nr, Nc = im.shape
    N = Nr * Nc
    labels_flat = labels_gt.flatten()

    # output directory
    out = f"output/{dataset}_img{image_index}_seed{seed}"
    os.makedirs(out, exist_ok=True)

    # plot original (natural only) + input
    if original_image is not None:
        plt.imsave(f"{out}/original.png", original_image, cmap="gray")
    plt.imsave(f"{out}/input.png", im, cmap="gray")

    # --- build ω and connectivity ---
    omega = im.flatten().astype(jnp.float64)
    B1s = vmap(gaussian_sheet, in_axes=(None,None,0,0))(Nr, Nc, alpha[0], sigma[0]).astype(jnp.complex128)
    B2s = vmap(gaussian_sheet, in_axes=(None,None,0,0))(Nr, Nc, alpha[1], sigma[1]).astype(jnp.complex128)

    # --- build ensemble of models ---
    @eqx.filter_vmap
    def make_model(key_i, B1_i, B2_i):
        l1 = CVRNNLayer(B=B1_i, nt=nt1, key=key_i)
        l2 = CVRNNLayer(B=B2_i, nt=nt2-nt1)
        return MultiLayerCVRNN([l1,l2])

    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, ensemble_size)
    models = make_model(keys, B1s, B2s)

    # --- run ensemble ---
    @eqx.filter_vmap(in_axes=(eqx.if_array(0), None))
    def run_ensemble(model, omega):
        return model(omega=omega)

    (h1s,h2s),(mask1s,mask2s) = run_ensemble(models, omega)

    # --- postprocess each member ---
    accs = []
    for i in range(ensemble_size):
        # reconstruct full history + NaNs
        h1,h2 = h1s[i], h2s[i]
        m1 = mask1s[i]
        combined = jnp.concatenate([h1,h2],axis=0)     # (nt2,N)
        nan_c    = jnp.array(0+0j).at[()].set(complex(jnp.nan,jnp.nan))
        tmask    = jnp.arange(nt2) >= nt1              # (nt2,)
        mask_time= jnp.outer(tmask,m1)                 # (nt2,N)
        combined = jnp.where(mask_time, nan_c, combined)

        X = combined.T                                 # (N,nt2)
        x_full = jnp.angle(X).reshape(Nr,Nc,-1)

        # optional dynamics plot, and only do for one of the seeds in the ensemble
        if visualize_dynamics and i == 0:
            plot_dynamics(x_full, nt1)

        # similarity projection
        x = jnp.exp(1j*x_full.reshape(-1,x_full.shape[2]))
        valid = ~m1
        x_valid = x[valid,:]
        rho,V,D,prj = spatiotemporal_segmentation(
            x_valid, dims, (nt1,nt2), window_size, window_step
        )
        w = prj.shape[-1]-1
        prj_last = prj[:,:,w]
        colors   = jnp.angle(x_valid[:,120])

        # plot 3D embedding
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            np.array(prj_last[:,0]),
            np.array(prj_last[:,1]),
            np.array(prj_last[:,2]),
            c=np.array(colors), cmap="hsv", s=50
        )
        ax.set_title("similarity projection")
        fig.colorbar(ax.collections[0], label="phase (rad)")
        fig.savefig(f"{out}/sim_{i}.png",bbox_inches="tight")
        plt.close(fig)

        # k-means + segmentation
        km = KMeans(n_clusters=n_clusters, random_state=seed)
        preds = km.fit_predict(np.array(prj_last[:,:3]))
        segmented = np.zeros(N,dtype=np.int32)
        segmented[np.where(valid)[0]] = preds+1
        segmented = segmented.reshape(Nr,Nc)
        plt.imsave(f"{out}/seg_{i}.png", segmented, cmap="viridis")

        # accuracy via Hungarian on true-foreground if requested
        y_pred = segmented.flatten().astype(np.int32)
        y_true = labels_flat.copy()
        if exclude_background:
            fg = (labels_flat!=0)
            y_pred = y_pred[fg]
            y_true = y_true[fg]

        P = np.unique(y_pred)
        T = np.unique(y_true)
        M = np.zeros((P.size,T.size),int)
        for pi,p in enumerate(P):
            for tj,t in enumerate(T):
                M[pi,tj] = np.sum((y_pred==p)&(y_true==t))
        row,col = linear_sum_assignment(M.max()-M)
        correct = M[row,col].sum()
        acc = correct / y_true.size
        accs.append(acc)

    # final summary
    accs = np.array(accs)
    print(f"Ensemble of {ensemble_size}: mean acc = {accs.mean():.4f}, std = {accs.std():.4f}")

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",          choices=["2shapes","3shapes","natural_image"],
                   required=True, help="which dataset")
    p.add_argument("--image_index",      type=int,   default=0)
    p.add_argument("--seed",             type=int,   default=1)
    p.add_argument("--ensemble_size",    type=int,   default=1)
    p.add_argument("--exclude_background", action="store_true")
    p.add_argument("--visualize_dynamics", action="store_true")
    args = p.parse_args()

    main(
      dataset           = args.dataset,
      image_index       = args.image_index,
      seed              = args.seed,
      ensemble_size     = args.ensemble_size,
      exclude_background= args.exclude_background,
      visualize_dynamics= args.visualize_dynamics,
    )
