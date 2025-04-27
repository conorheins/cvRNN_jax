import jax
import jax.numpy as jnp
import numpy as np
from cv_rnn import gaussian_sheet
from cvrnn_layer import CVRNNLayer
import unittest
from scipy.io import loadmat

class TestCVRNNLayer(unittest.TestCase):
    
    def test_cvrnn_layer_matches_original(self):
        # Set the same random seed for reproducibility
        seed = 42
        key = jax.random.PRNGKey(seed)

        dataset_names = ['2shapes', '3shapes', 'natural_image']

        a, s = 0.5, 0.9  # Connection parameters
        nt = 60          # Number of timesteps        

        for dataset_name in dataset_names:
            data = loadmat(f'dataset/{dataset_name}.mat')

            if dataset_name == "natural_image":
                im = data["im"]
            else:
                im = data['images'][:, :, 0]   # get first image

            # --- hyperparameters ---
            Nr, Nc = im.shape
        
            # Build the connection matrix (same as in run_2layer)
            K1 = gaussian_sheet(Nr, Nc, a, s)
            K1 = K1.astype(jnp.complex128)
            
            # Generate random initial condition
            key, x0_key = jax.random.split(key)
            rand_angles = jax.random.uniform(x0_key, shape=(Nr * Nc,), minval=-jnp.pi, maxval=jnp.pi)
            x0 = jnp.exp(1j * rand_angles)
            
            # Generate random omega (frequencies)
            key, omega_key = jax.random.split(key)
            omega = jax.random.uniform(omega_key, shape=(Nr * Nc,), minval=0.0, maxval=1.0)
            omega = omega.astype(jnp.float64)
            
            # Run the original implementation
            def step_fn(carry, _):
                x = carry
                x = (1j * omega) * x + K1 @ x
                return x, x

            _, x_history_original = jax.lax.scan(step_fn, x0, None, length=nt-1)
            x_history_original = jnp.concatenate((x0[None, :], x_history_original), axis=0)
            
            # Test 1: Pass x0 explicitly in the call
            layer1 = CVRNNLayer(B=K1, nt=nt)
            x_history1 = layer1(omega=omega, x0=x0)
            
            # Test 2: Initialize with x0 during construction
            layer2 = CVRNNLayer(B=K1, nt=nt, x0=x0)
            x_history2 = layer2(omega=omega)
            
            # Test 3: Initialize with the same random key that was used to generate x0
            layer3 = CVRNNLayer(B=K1, nt=nt, key=x0_key)
            x_history3 = layer3(omega=omega)
            
            # Test 4: Use the key at runtime in __call__
            layer4 = CVRNNLayer(B=K1, nt=nt)  # Empty layer
            x_history4 = layer4(omega=omega, key=x0_key)  # Generate x0 at runtime with same key
            
            # Compare all results with the original
            np.testing.assert_allclose(
                x_history_original, 
                x_history1, 
                rtol=1e-10, atol=1e-10,
                err_msg=f"Method 1: Explicit x0 in call doesn't match original implementation for {dataset_name} dataset"
            )
            
            np.testing.assert_allclose(
                x_history_original, 
                x_history2, 
                rtol=1e-10, atol=1e-10,
                err_msg=f"Method 2: x0 set during initialization doesn't match original implementation for {dataset_name} dataset"
            )
            
            np.testing.assert_allclose(
                x_history_original, 
                x_history3, 
                rtol=1e-10, atol=1e-10,
                err_msg=f"Method 3: Using same initialization key doesn't match original implementation for {dataset_name} dataset"
            )
            
            np.testing.assert_allclose(
                x_history_original, 
                x_history4, 
                rtol=1e-10, atol=1e-10,
                err_msg=f"Method 4: Using key at runtime in __call__ doesn't match original implementation for {dataset_name} dataset"
            )
            
            print(f"✓ First CVRNNLayer produces identical results to the original implementation for {dataset_name} dataset")
            
    def test_random_initialization(self):
        # Test initialization with a random key
        Nr, Nc = 5, 5
        a, s = 0.8, 0.2
        nt = 10
        
        # Build the connection matrix
        K = gaussian_sheet(Nr, Nc, a, s).astype(jnp.complex128)
        
        # Create omega
        omega = jnp.ones((Nr * Nc,), dtype=jnp.float64) * 0.5
        
        # Initialize with a random key
        key = jax.random.PRNGKey(123)
        layer = CVRNNLayer(B=K, nt=nt, key=key)
        
        # Make sure we can run the layer with the internally generated x0
        output = layer(omega=omega)
        
        # Verify shape is correct
        self.assertEqual(output.shape[0], nt)
        self.assertEqual(output.shape[1], Nr * Nc)
        
        # Verify complex dtype
        self.assertEqual(output.dtype, jnp.complex128)
        
        # Test runtime dynamic initialization (key passed to __call__)
        key1 = jax.random.PRNGKey(456)
        key2 = jax.random.PRNGKey(789)
        
        # Same layer, different keys at runtime should give different results
        output1 = layer(omega=omega, key=key1)
        output2 = layer(omega=omega, key=key2)
        
        # Results should be different
        are_equal = jnp.allclose(output1, output2, rtol=1e-10, atol=1e-10)
        self.assertFalse(are_equal, "Using different keys at runtime should produce different results")
        
        # Same key at runtime should give same results
        output3 = layer(omega=omega, key=key1)
        output4 = layer(omega=omega, key=key1)
        
        # Results should be identical
        np.testing.assert_allclose(
            output3, 
            output4, 
            rtol=1e-10, atol=1e-10,
            err_msg="Using the same key at runtime should produce identical results"
        )
        
        print("✓ Random initialization tests passed")


if __name__ == "__main__":
    unittest.main() 