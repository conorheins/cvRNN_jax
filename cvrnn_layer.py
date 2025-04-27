import equinox as eqx
import jax, jax.numpy as jnp
from typing import Optional, Sequence, Tuple

class CVRNNLayer(eqx.Module):
    """
    One cv-RNN layer:
      - B:  (N×N) complex connectivity
      - nt: timesteps
      - x0: optional stored initial state
    """
    B: jnp.ndarray
    nt: int
    x0: Optional[jnp.ndarray]

    def __init__(self,
                 B: jnp.ndarray,
                 nt: int,
                 x0: Optional[jnp.ndarray] = None,
                 key: Optional[jax.Array] = None):
        """
        Initialize a CV-RNN layer.
        
        Parameters
        ----------
        B : complex connectivity matrix, shape (N, N)
        nt : number of timesteps to unroll
        x0 : optional pre-defined initial state, shape (N,)
        key : optional random key for generating x0 if not provided
        """
        # B should already be complex128 of shape (N, N)
        self.B = B
        self.nt = nt
        # Initialize x0 if provided, otherwise keep as None
        if x0 is not None:
            self.x0 = x0
        elif key is not None:
            # Generate x0 from key
            N = B.shape[0]
            rand_angles = jax.random.uniform(
                key, shape=(N,), minval=-jnp.pi, maxval=jnp.pi
            )
            self.x0 = jnp.exp(1j * rand_angles)
        else:
            self.x0 = None

    def _generate_x0(self, key: jax.Array) -> jnp.ndarray:
        """Generate random initial condition from a key."""
        N = self.B.shape[0]
        rand_angles = jax.random.uniform(
            key, shape=(N,), minval=-jnp.pi, maxval=jnp.pi
        )
        return jnp.exp(1j * rand_angles)

    def __call__(self,
                 omega: jnp.ndarray,                  # shape (..., N), float64
                 x0: Optional[jnp.ndarray] = None,    # shape (..., N), complex128
                 key: Optional[jax.Array] = None,     # optional key for runtime initialization
                 mask: Optional[jnp.ndarray] = None,  # boolean shape (..., N)
                 include_initial: bool = True         # whether to prepend initial state
                 ) -> jnp.ndarray:
        """
        Runs the recurrent dynamics for nt steps.

        Parameters
        ----------
        omega : frequencies for each node, shape (..., N)
        x0 : optional initial state, shape (..., N). If None, uses the internally stored x0.
        key : optional random key to generate x0 at runtime (highest priority)
        
        Returns
        -------
        x_history : complex128 array, shape (nt, ..., N)
        
        Notes
        -----
        Priority for determining initial state:
        1. If key is provided, generate new x0 at runtime
        2. If x0 is provided directly to __call__, use that 
        3. If self.x0 exists (set during init), use that
        4. Otherwise, raise ValueError
        """
        # Determine the initial state to use, with priority order
        if key is not None:
            # Generate new x0 at runtime (highest priority)
            initial_x = self._generate_x0(key)
        elif x0 is not None:
            # Use explicitly provided x0
            initial_x = x0
        elif self.x0 is not None:
            # Use the stored x0 from initialization
            initial_x = self.x0
        else:
            # No initial state available
            raise ValueError("No initial state (x0) provided. Either pass x0 or key to __call__, or initialize the layer with x0 or key.")


        # 2) apply mask if present
        if mask is not None:
            # mask == True ⇒ background ⇒ zero out
            valid_f = (~mask).astype(omega.dtype)
            B_eff = self.B * jnp.outer(valid_f, valid_f)
            omega_eff = jnp.where(mask, 0.0, omega)
            initial_x = jnp.where(mask, 0.0 + 0j, initial_x)
        else:
            B_eff = self.B
            omega_eff = omega

        # 3) unroll
        def step(x, _):
            # x: shape (..., N)
            # the core update: x ← (i·ω)*x + B @ x
            x_next = (1j * omega_eff) * x + jnp.matmul(B_eff, x[..., None])[..., 0]
            return x_next, x_next

        if include_initial:
            # scan nt-1 steps, then prepend initial_x, total length = nt
            _, hist = jax.lax.scan(step, initial_x, None, length=self.nt - 1)
            hist = jnp.concatenate([initial_x[None, ...], hist], axis=0)
        else:
            # scan exactly nt steps, return the post-step states x1…x_nt
            _, hist = jax.lax.scan(step, initial_x, None, length=self.nt)
        return hist


class MultiLayerCVRNN(eqx.Module):
    """
    Stack of CVRNNLayer. On each layer:
      - run layer_i from the SAME original x0 but masked by mask_{i-1}
      - compute mask_i from the final-phase mean‐threshold rule
    """
    layers: Tuple[CVRNNLayer, ...]
    
    def __init__(self, layers: Sequence[CVRNNLayer]):
        self.layers = tuple(layers)

    def __call__(self,
                 omega: jnp.ndarray,        # shape (N,)
                 x0: Optional[jnp.ndarray] = None,
                 key: Optional[jax.Array] = None
                 ) -> Tuple[Tuple[jnp.ndarray, ...], Tuple[jnp.ndarray, ...]]:
        # Determine the original x0 once (for all layers)
        if key is not None:
            # use first layer’s generator
            x0_orig = self.layers[0]._generate_x0(key)
        elif x0 is not None:
            x0_orig = x0
        elif self.layers[0].x0 is not None:
            x0_orig = self.layers[0].x0
        else:
            raise ValueError("Must supply x0 or key")

        histories = []
        masks     = []
        prev_mask = None

        for i, layer in enumerate(self.layers):
            # include initial only in layer 0; layer 1+ use only post-step history
            h = layer(
                omega=omega,
                x0=x0_orig,
                mask=prev_mask,
                include_initial=(i == 0),
            )
            histories.append(h)

            # compute new mask from final phase
            final_phase = jnp.angle(h[-1])
            thr = jnp.mean(final_phase)
            above = final_phase > thr
            below = final_phase < thr
            # choose the smaller‐group rule
            count_above = jnp.sum(above)
            count_below = jnp.sum(below)
            mask_i = jax.lax.cond(count_above > count_below,
                                  lambda _: above,
                                  lambda _: below,
                                  operand=None)
            masks.append(mask_i)
            prev_mask = mask_i

        return tuple(histories), tuple(masks)
