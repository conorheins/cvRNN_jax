import equinox as eqx
import jax, jax.numpy as jnp
from typing import Optional, Tuple

class CVRNNLayer(eqx.Module):
    """
    One cv-RNN layer:
      - B:      (N×N) complex connectivity matrix
      - nt:     number of timesteps to unroll
      - x0:     optional fixed initial condition
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
                 omega: jnp.ndarray,     # shape (..., N), float64
                 x0: Optional[jnp.ndarray] = None,  # shape (..., N), complex128
                 key: Optional[jax.Array] = None    # optional key for runtime initialization
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

        def step(x, _):
            # x: shape (..., N)
            # the core update: x ← (i·ω)*x + B @ x
            x_next = (1j * omega) * x + jnp.matmul(self.B, x[..., None])[..., 0]
            return x_next, x_next

        # lax.scan will broadcast over any leading batch dims in initial_x/omega
        _, hist = jax.lax.scan(step, initial_x, None, length=self.nt - 1)

        # prepend the initial state
        return jnp.concatenate([initial_x[None, ...], hist], axis=0)
