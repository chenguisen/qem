import os
from abc import ABC, abstractmethod
from functools import partial

import numpy as np
from dotenv import load_dotenv
from numba import jit as njit

load_dotenv()
import keras
from keras import ops
class ImageModel(keras.Model):
    """Base class for all image models."""

    def __init__(self, dx: float=1.0):
        """Initialize the model.
        
        Args:
            dx (float, optional): Pixel size. Defaults to 1.0.
        """
        super().__init__()
        self.dx = dx
        self.input_params = None
        self.x_grid= None
        self.y_grid = None
        
    def set_grid(self, x_grid, y_grid):
        self.x_grid = x_grid
        self.y_grid = y_grid
            
    def set_params(self, params):
        # Set params as tensors, but do not build variables yet
        self.input_params = {k: keras.ops.convert_to_tensor(v) for k, v in params.items()}
        # If already built and shapes match, update values
        if self.built:
            self.update_params(self.input_params)

    def update_params(self, params):
        """Update the model parameters (values only, not shapes)."""
        for key, value in params.items():
            if hasattr(self, key):
                current_value = getattr(self, key)
                if isinstance(current_value, keras.Variable):
                    current_value.assign(value)
                else:
                    setattr(self, key, value)
            else:
                raise ValueError(f"Parameter {key} does not exist in the model.")

    def build(self, input_shape):
        if self.input_params is None:
            raise ValueError("initial_params must be set before building the model.")
        # If already built and shapes match, do nothing
        if self.built:
            return
        # Otherwise, create new variables
        self.pos_x = self.add_weight(shape=(self.input_params['pos_x'].shape[0],), initializer=keras.initializers.Constant(self.input_params['pos_x']), name="pos_x")
        self.pos_y = self.add_weight(shape=(self.input_params['pos_y'].shape[0],), initializer=keras.initializers.Constant(self.input_params['pos_y']), name="pos_y")
        self.height = self.add_weight(shape=(self.input_params['height'].shape[0],), initializer=keras.initializers.Constant(self.input_params['height']), name="height")
        self.width = self.add_weight(shape=(self.input_params['width'].shape[0],), initializer=keras.initializers.Constant(self.input_params['width']), name="width")
        self.background = self.add_weight(shape=(), initializer=keras.initializers.Constant(self.input_params['background']), name="background")
        super().build(input_shape)
        
    def get_params(self):
        return {
            "pos_x": keras.ops.convert_to_tensor(self.pos_x),
            "pos_y": keras.ops.convert_to_tensor(self.pos_y),
            "height": keras.ops.convert_to_tensor(self.height),
            "width": keras.ops.convert_to_tensor(self.width),
            "background": keras.ops.convert_to_tensor(self.background),
        }

    def call(self, local):
        """Forward pass of the model."""
        return self.sum(local=local)

    @abstractmethod
    def model_fn(self, x, y, pos_x, pos_y, height, width, *args):
        """Core model function that defines the peak shape."""
        pass


    @abstractmethod
    def volume(self, params: dict) -> np.ndarray:
        """Calculate the volume of each peak."""
        pass

    def _sum(self,  local=True):
        """Calculate all peaks either globally or locally.
        
        Args:
            local (bool, optional): If True, calculate peaks locally within a fixed window. Defaults to False.
            
        Returns:
            array: Sum of all peaks plus background
        """
        kargs = []
        if hasattr(self, 'ratio'):
            kargs.append(self.ratio)

        if not local:
            # Calculate all peaks at once and sum them
            peaks = self.model_fn(
                self.x_grid[:, :, None], self.y_grid[:, :, None],
                self.pos_x[None, None, :], self.pos_y[None, None, :],
                self.height, self.width, *kargs
            )
            return keras.ops.sum(peaks, axis=-1) + self.background
        else:
            # Local calculation with parallel processing
            width_max = keras.ops.max(self.width) 
            window_size = keras.ops.cast(keras.ops.ceil(width_max * 4), dtype="int32")

            # Create a local coordinate grid for the window
            window_x = keras.ops.arange(-window_size, window_size + 1, dtype=self.x_grid.dtype)
            window_y = keras.ops.arange(-window_size, window_size + 1, dtype=self.y_grid.dtype)
            local_x_grid, local_y_grid = keras.ops.meshgrid(window_x, window_y)

            # Calculate local peaks relative to their centers (0,0)
            # The positions are implicitly handled by where we add the peaks back.
            input_params = (keras.ops.mod(self.pos_x, 1), keras.ops.mod(self.pos_y, 1), self.height, self.width)
            if hasattr(self, 'ratio'):
                input_params += (self.ratio,)
            
            peak_local = self.model_fn(local_x_grid[..., None], local_y_grid[..., None], *input_params)

            # Calculate integer base coordinates for each peak
            pos_x_int = keras.ops.floor(self.pos_x)
            pos_y_int = keras.ops.floor(self.pos_y)

            # Calculate the global coordinates for each point in each local peak window
            global_x = keras.ops.expand_dims(local_x_grid, -1) + pos_x_int
            global_y = keras.ops.expand_dims(local_y_grid, -1) + pos_y_int

            # Create a mask for coordinates that are within the image boundaries
            mask = (global_x >= 0) & (global_x < self.x_grid.shape[1]) & (global_y >= 0) & (global_y < self.y_grid.shape[0])

            # Get the indices of valid elements where the mask is True.
            valid_indices = keras.ops.where(mask)
            
            # Flatten the mask to get 1D indices of valid elements.
            flat_indices = keras.ops.where(keras.ops.reshape(mask, (-1,)))[0]

            # Gather the valid data from the flattened tensors using the 1D indices.
            valid_values = keras.ops.take(keras.ops.reshape(peak_local, (-1,)), flat_indices)
            global_x_valid = keras.ops.take(keras.ops.reshape(global_x, (-1,)), flat_indices)
            global_y_valid = keras.ops.take(keras.ops.reshape(global_y, (-1,)), flat_indices)

            # Create the final image tensor
            total = keras.ops.zeros_like(self.x_grid, dtype=self.x_grid.dtype)
            
            # Use the backend to scatter the local peaks onto the global image
            backend = keras.backend.backend()
            if backend == 'torch':
                import torch
                # Calculate flat indices for scatter_add
                indices = (keras.ops.cast(global_y_valid, dtype='int64') * total.shape[1] + keras.ops.cast(global_x_valid, dtype='int64'))
                
                total_flat = total.flatten()
                # Use a non-in-place operation to help PyTorch's autograd manage memory
                total_flat = total_flat.scatter_add(0, indices, valid_values)
                total = total_flat.reshape(total.shape)
            elif backend == 'jax':
                import jax.numpy as jnp
                # JAX uses a different approach for indexed updates
                indices = (keras.ops.cast(global_y_valid, 'int32'), keras.ops.cast(global_x_valid, 'int32'))
                total = total.at[indices].add(valid_values)
            else: # tensorflow
                import tensorflow as tf
                # TensorFlow uses tensor_scatter_nd_add
                indices = keras.ops.stack([keras.ops.cast(global_y_valid, dtype='int32'), keras.ops.cast(global_x_valid, dtype='int32')], axis=-1)
                total = tf.tensor_scatter_nd_add(total, indices, valid_values)

            return total + self.background

    def sum(self, local=True):
        """Calculate sum of peaks using Keras.
        
        Args:
            x_grid (array): x_grid coordinates mesh
            y_grid (array): y_grid coordinates mesh
            local (bool, optional): If True, calculate peaks locally within a fixed window. Defaults to False.
            
        Returns:
            array: Sum of all peaks plus background
        """
        return self._sum(local=local)

class GaussianModel(ImageModel):
    """Gaussian peak model."""

    def volume(self, params: dict) -> np.ndarray:
        """Calculate the volume of each Gaussian peak.
        
        For a 2D Gaussian, the volume is: height * 2π * width²
        """
        height = params["height"]
        width = params["width"]
        return height * 2 * np.pi * width**2 * self.dx**2

    def model_fn(self, x, y, pos_x, pos_y, height, width, *args):
        """Core computation for Gaussian model using Keras."""
        return height * keras.ops.exp(
            -(keras.ops.square(x - pos_x) + keras.ops.square(y - pos_y)) / (2 * keras.ops.square(width))
        )

class LorentzianModel(ImageModel):
    """Lorentzian peak model."""

    def volume(self, params: dict) -> np.ndarray:
        """Calculate the volume of each Lorentzian peak.
        
        For a 2D Lorentzian, the volume is: height * π * width²
        """
        height = params["height"]
        width = params["width"]
        return height * np.pi * width**2 * self.dx**2

    def model_fn(self, x, y, pos_x, pos_y, height, width, *args):
        """Core computation for Lorentzian model using Keras."""
        return height / (
            1 + (keras.ops.square(x - pos_x) + keras.ops.square(y - pos_y)) / keras.ops.square(width)
        )


class VoigtModel(ImageModel):
    """Voigt peak model."""

    def __init__(self, dx: float=1.0, background: float=0.0):
        """Initialize the model.
        
        Args:
            dx (float, optional): Pixel size. Defaults to 1.0.
            background (float, optional): Background level. Defaults to 0.0.
        """
        super().__init__(dx, background)
        # self.model_type = "voigt"

    def set_params(self, params):
        super().set_params(params)
        self.ratio = self.input_params['ratio']

    def build(self, input_shape):
        super().build(input_shape)
        self.ratio = self.add_weight(shape=(), initializer=keras.initializers.Constant(self.input_params['ratio']), name="ratio")

    def get_params(self):
        params = super().get_params()
        params['ratio'] = keras.ops.convert_to_tensor(self.ratio)
        return params

    def volume(self, params: dict) -> np.ndarray:
        """Calculate the volume of each Voigt peak.
        
        For a 2D Voigt profile, the volume is a weighted sum of Gaussian and Lorentzian volumes:
        V = ratio * (height * 2π * width²) + (1-ratio) * (height * π * width²)
        """
        height = params["height"]
        width = params["width"]
        ratio = params["ratio"]
        
        gaussian_vol = height * 2 * np.pi * width**2 * self.dx**2
        lorentzian_vol = height * np.pi * width**2 * self.dx**2
        
        return ratio * gaussian_vol + (1 - ratio) * lorentzian_vol

    def model_fn(self, x, y, pos_x, pos_y, height, width, ratio):
        """Core computation for Voigt model using Keras."""
        # Convert width to sigma and gamma
        sigma = width
        gamma = width / keras.ops.sqrt(2 * keras.ops.log(2.0))
        
        # Calculate squared distance
        r2 = keras.ops.square(x - pos_x) + keras.ops.square(y - pos_y)
        
        # Compute Gaussian and Lorentzian parts
        gaussian_part = keras.ops.exp(-r2 / (2 * sigma**2))
        lorentzian_part = gamma**3 / keras.ops.power(r2 + gamma**2, 3/2)
        
        # Return weighted sum
        return height * (ratio * gaussian_part + (1 - ratio) * lorentzian_part)


class GaussianKernel:
    """Gaussian kernel implementation."""

    def __init__(self):
        """Initialize the kernel.
        
        Args:
            backend (str, optional): Backend to use ('tensorflow', 'pytorch', or 'jax'). Defaults to 'jax'.
        """

    def gaussian_kernel(self, sigma):
        """Creates a 2D Gaussian kernel with the given sigma."""
        size = int(4 * sigma + 0.5) * 2 + 1  # Odd size
        x = keras.ops.arange(-(size // 2), (size // 2) + 1, dtype='float32')
        x_grid, y_grid = keras.ops.meshgrid(x, x)
        kernel = keras.ops.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
        return kernel / keras.ops.sum(kernel)

    def gaussian_filter(self, image, sigma):
        """Applies Gaussian filter to a 2D image."""
        # Ensure both image and kernel are float32
        image = keras.ops.cast(image, 'float32')
        kernel = self.gaussian_kernel(sigma)
        # Add channel dimensions for input and kernel
        image = keras.ops.expand_dims(keras.ops.expand_dims(image, 0), -1)  # [1, H, W, 1]
        kernel = keras.ops.expand_dims(keras.ops.expand_dims(kernel, -1), -1)  # [H, W, 1, 1]
        filtered = keras.ops.conv(image, kernel, padding='same')
        return keras.ops.squeeze(filtered)  # Remove extra dimensions

@njit
def gaussian_2d_single(xy, pos_x, pos_y, height, width, background):
    """2D Gaussian function for single atom."""
    x_grid, y_grid = xy
    return (
        height
        * np.exp(
            -((x_grid[:,:,None] - pos_x) ** 2 + (y_grid[:,:,None] - pos_y) ** 2) / (2 * width**2)
        ) + background
    ).ravel()