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

    def __init__(self, dx: float=1.0, background: float=0.0):
        """Initialize the model.
        
        Args:
            dx (float, optional): Pixel size. Defaults to 1.0.
            background (float, optional): Background level. Defaults to 0.0.
        """
        super().__init__()
        self.dx = dx
        self.background = background
        ops = keras.ops
            
    def set_params(self, params):
        self.initial_params = {k: ops.convert_to_tensor(v) for k, v in params.items()}
        self.num_coordinates = self.initial_params['pos_x'].shape[0]
        
    def build(self, input_shape):
        """Create and initialize trainable weights for the model."""
        self.pos_x = self.add_weight(shape=(self.initial_params['pos_x'].shape[0],), initializer=keras.initializers.Constant(self.initial_params['pos_x']), name="pos_x")
        self.pos_y = self.add_weight(shape=(self.initial_params['pos_y'].shape[0],), initializer=keras.initializers.Constant(self.initial_params['pos_y']), name="pos_y")
        self.height = self.add_weight(shape=(self.initial_params['height'].shape[0],), initializer=keras.initializers.Constant(self.initial_params['height']), name="height")
        self.width = self.add_weight(shape=(self.initial_params['width'].shape[0],), initializer=keras.initializers.Constant(self.initial_params['width']), name="width")
        self.background = self.add_weight(shape=(), initializer=keras.initializers.Constant(self.initial_params['background']), name="background")
        super().build(input_shape)
        
    def get_params(self):
        return {
            "pos_x": ops.convert_to_tensor(self.pos_x),
            "pos_y": ops.convert_to_tensor(self.pos_y),
            "height": ops.convert_to_tensor(self.height),
            "width": ops.convert_to_tensor(self.width),
            "background": ops.convert_to_tensor(self.background),
        }

    def call(self, inputs):
        """Forward pass of the model."""
        x_grid, y_grid = inputs
        return self.sum(x_grid, y_grid)

    @abstractmethod
    def model_fn(self, x, y, pos_x, pos_y, height, width, *args):
        """Core model function that defines the peak shape."""
        pass


    @abstractmethod
    def volume(self, params: dict) -> np.ndarray:
        """Calculate the volume of each peak."""
        pass

    def _sum(self, x_grid, y_grid, local=True):
        """Calculate all peaks either globally or locally.
        
        Args:
            x_grid (array): x_grid coordinates mesh
            y_grid (array): y_grid coordinates mesh
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
                x_grid[:, :, None], y_grid[:, :, None],
                self.pos_x[None, None, :], self.pos_y[None, None, :],
                self.height, self.width, *kargs
            )
            return ops.sum(peaks, axis=-1) + self.background
        else:
            # Local calculation with parallel processing
            width_max = ops.max(self.width) 
            window_size = ops.cast(ops.ceil(width_max * 4), dtype="int32")

            # Create a local coordinate grid for the window
            window_x = ops.arange(-window_size, window_size + 1, dtype=x_grid.dtype)
            window_y = ops.arange(-window_size, window_size + 1, dtype=y_grid.dtype)
            local_x_grid, local_y_grid = ops.meshgrid(window_x, window_y)

            # Calculate local peaks relative to their centers (0,0)
            # The positions are implicitly handled by where we add the peaks back.
            input_params = (ops.mod(self.pos_x, 1), ops.mod(self.pos_y, 1), self.height, self.width)
            if hasattr(self, 'ratio'):
                input_params += (self.ratio,)
            
            peak_local = self.model_fn(local_x_grid[..., None], local_y_grid[..., None], *input_params)

            # Calculate integer base coordinates for each peak
            pos_x_int = ops.floor(self.pos_x)
            pos_y_int = ops.floor(self.pos_y)

            # Calculate the global coordinates for each point in each local peak window
            global_x = ops.expand_dims(local_x_grid, -1) + pos_x_int
            global_y = ops.expand_dims(local_y_grid, -1) + pos_y_int

            # Create a mask for coordinates that are within the image boundaries
            mask = (global_x >= 0) & (global_x < x_grid.shape[1]) & (global_y >= 0) & (global_y < y_grid.shape[0])

            # Get the indices of valid elements where the mask is True.
            valid_indices = ops.where(mask)
            
            # Flatten the mask to get 1D indices of valid elements.
            flat_indices = ops.where(ops.reshape(mask, (-1,)))[0]

            # Gather the valid data from the flattened tensors using the 1D indices.
            valid_values = ops.take(ops.reshape(peak_local, (-1,)), flat_indices)
            global_x_valid = ops.take(ops.reshape(global_x, (-1,)), flat_indices)
            global_y_valid = ops.take(ops.reshape(global_y, (-1,)), flat_indices)

            # The column indices correspond to the third dimension of the original tensors.
            cols_tensor = valid_indices[2]

            # Create the final image tensor
            total = ops.zeros_like(x_grid, dtype=x_grid.dtype)
            
            # Use the backend to scatter the local peaks onto the global image
            backend = keras.backend.backend()
            if backend == 'torch':
                import torch
                # Calculate flat indices for scatter_add
                indices = (ops.cast(global_y_valid, dtype='int64') * total.shape[1] + ops.cast(global_x_valid, dtype='int64'))
                
                total_flat = total.flatten()
                # Use a non-in-place operation to help PyTorch's autograd manage memory
                total_flat = total_flat.scatter_add(0, indices, valid_values)
                total = total_flat.reshape(total.shape)
            elif backend == 'jax':
                import jax.numpy as jnp
                # JAX uses a different approach for indexed updates
                indices = (ops.cast(global_y_valid, 'int32'), ops.cast(global_x_valid, 'int32'))
                total = total.at[indices].add(valid_values)
            else: # tensorflow
                import tensorflow as tf
                # TensorFlow uses tensor_scatter_nd_add
                indices = ops.stack([ops.cast(global_y_valid, dtype='int32'), ops.cast(global_x_valid, dtype='int32')], axis=-1)
                total = tf.tensor_scatter_nd_add(total, indices, valid_values)

            return total + self.background

    def sum(self, x_grid, y_grid, local=False):
        """Calculate sum of peaks using Keras.
        
        Args:
            x_grid (array): x_grid coordinates mesh
            y_grid (array): y_grid coordinates mesh
            local (bool, optional): If True, calculate peaks locally within a fixed window. Defaults to False.
            
        Returns:
            array: Sum of all peaks plus background
        """
        return self._sum(x_grid, y_grid, local=local)

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
        return height * ops.exp(
            -(ops.square(x - pos_x) + ops.square(y - pos_y)) / (2 * ops.square(width))
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
            1 + (ops.square(x - pos_x) + ops.square(y - pos_y)) / ops.square(width)
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
        self.ratio = self.initial_params['ratio']

    def build(self, input_shape):
        super().build(input_shape)
        self.ratio = self.add_weight(shape=(), initializer=keras.initializers.Constant(self.initial_params['ratio']), name="ratio")

    def get_params(self):
        params = super().get_params()
        params['ratio'] = ops.convert_to_tensor(self.ratio)
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
        gamma = width / ops.sqrt(2 * ops.log(2.0))
        
        # Calculate squared distance
        r2 = ops.square(x - pos_x) + ops.square(y - pos_y)
        
        # Compute Gaussian and Lorentzian parts
        gaussian_part = ops.exp(-r2 / (2 * sigma**2))
        lorentzian_part = gamma**3 / ops.power(r2 + gamma**2, 3/2)
        
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
        x = ops.arange(-(size // 2), (size // 2) + 1, dtype='float32')
        x_grid, y_grid = ops.meshgrid(x, x)
        kernel = ops.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
        return kernel / ops.sum(kernel)

    def gaussian_filter(self, image, sigma):
        """Applies Gaussian filter to a 2D image."""
        # Ensure both image and kernel are float32
        image = ops.cast(image, 'float32')
        kernel = self.gaussian_kernel(sigma)
        # Add channel dimensions for input and kernel
        image = ops.expand_dims(ops.expand_dims(image, 0), -1)  # [1, H, W, 1]
        kernel = ops.expand_dims(ops.expand_dims(kernel, -1), -1)  # [H, W, 1, 1]
        filtered = ops.conv(image, kernel, padding='same')
        return ops.squeeze(filtered)  # Remove extra dimensions

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