"""Backend detection and configuration utilities."""
import os
import warnings


def detect_available_backends():
    """
    Detect which Keras backends are available in the current environment.
    
    Returns:
        list: List of available backend names in order of preference
    """
    available_backends = []
    
    # Check for JAX
    try:
        import jax
        import jaxlib
        available_backends.append('jax')
    except ImportError:
        pass
    
    # Check for PyTorch
    try:
        import torch
        available_backends.append('torch')
    except ImportError:
        pass
    
    # Check for TensorFlow
    try:
        import tensorflow
        available_backends.append('tensorflow')
    except ImportError:
        pass
    
    return available_backends


def get_best_backend():
    """
    Get the best available backend for the current environment.
    
    Returns:
        str: Name of the best available backend
        
    Raises:
        RuntimeError: If no backends are available
    """
    available = detect_available_backends()
    
    if not available:
        raise RuntimeError(
            "No Keras backends available. Please install at least one of: "
            "jax, torch, or tensorflow"
        )
    
    # Preference order: JAX > PyTorch > TensorFlow
    preference_order = ['jax', 'torch', 'tensorflow']
    
    for backend in preference_order:
        if backend in available:
            return backend
    
    # Fallback to first available
    return available[0]


def configure_backend(backend_name=None, force=False):
    """
    Configure Keras to use the specified backend.
    
    Args:
        backend_name (str, optional): Backend to use. If None, auto-detect best.
        force (bool): Whether to force reconfiguration even if already set.
        
    Returns:
        str: Name of the configured backend
    """
    if backend_name is None:
        backend_name = get_best_backend()
    
    # Check if backend is available
    available = detect_available_backends()
    if backend_name not in available:
        raise ValueError(
            f"Backend '{backend_name}' is not available. "
            f"Available backends: {available}"
        )
    
    # Set environment variable
    current_backend = os.environ.get("KERAS_BACKEND")
    if current_backend != backend_name or force:
        os.environ["KERAS_BACKEND"] = backend_name
        
        # Clear any existing Keras session
        try:
            import keras
            keras.backend.clear_session()
        except ImportError:
            pass
    
    return backend_name


def setup_test_backend():
    """
    Set up the best available backend for testing.
    
    Returns:
        str: Name of the configured backend
    """
    try:
        backend = configure_backend()
        print(f"Using Keras backend: {backend}")
        return backend
    except Exception as e:
        print(f"Warning: Failed to configure backend: {e}")
        # Try to use whatever is available
        available = detect_available_backends()
        if available:
            backend = available[0]
            os.environ["KERAS_BACKEND"] = backend
            print(f"Fallback to: {backend}")
            return backend
        return None


def backend_specific_config(backend_name):
    """
    Apply backend-specific configurations.
    
    Args:
        backend_name (str): Name of the backend to configure
    """
    if backend_name == 'jax':
        try:
            import jax
            # Force JAX to use CPU if no GPU available
            jax.config.update('jax_platforms', 'cpu')
            # Enable 64-bit precision for better numerical accuracy
            jax.config.update("jax_enable_x64", True)
        except (ImportError, Exception):
            pass
    
    elif backend_name == 'torch':
        try:
            import torch
            # Set default tensor type to float32
            torch.set_default_dtype(torch.float32)
            # Use CPU if CUDA is not available
            if not torch.cuda.is_available():
                torch.set_default_device('cpu')
        except (ImportError, Exception):
            pass
    
    elif backend_name == 'tensorflow':
        try:
            import tensorflow as tf
            # Suppress TensorFlow warnings
            tf.get_logger().setLevel('ERROR')
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            # Use CPU if GPU is not available
            tf.config.set_visible_devices([], 'GPU')
        except (ImportError, Exception):
            pass


# Auto-configure on import only if explicitly requested
def auto_configure():
    """Auto-configure backend if none is set."""
    try:
        if not os.environ.get("KERAS_BACKEND"):
            available = detect_available_backends()
            if available:
                backend = available[0]  # Use first available
                os.environ["KERAS_BACKEND"] = backend
                backend_specific_config(backend)
                return backend
    except Exception:
        pass
    return None

# Only auto-configure if this module is run directly
if __name__ == "__main__":
    auto_configure()