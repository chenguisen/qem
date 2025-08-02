#!/usr/bin/env python3
"""Minimal test to check if the model building issue is fixed."""

try:
    # Configure backend
    from qem.backend_utils import setup_test_backend
    setup_test_backend()
    
    import numpy as np
    from qem.image_fitting import ImageFitting
    
    print("Creating synthetic image...")
    # Create a simple synthetic image
    size = 20
    image = np.zeros((size, size))
    
    # Add a simple peak manually
    center = size // 2
    for i in range(size):
        for j in range(size):
            r2 = (i - center)**2 + (j - center)**2
            image[i, j] = np.exp(-r2 / (2 * 2**2)) + 0.1
    
    print("Initializing ImageFitting...")
    # Initialize ImageFitting
    fitter = ImageFitting(
        image=image,
        dx=1.0,
        model_type="gaussian"
    )
    
    print("Setting coordinates...")
    # Set a single coordinate
    fitter.coordinates = np.array([[center, center]], dtype=float)
    
    print("Initializing parameters...")
    # Initialize parameters
    params = fitter.init_params(atom_size=2.0)
    print("✓ Parameters initialized successfully")
    
    print("Testing prediction...")
    # Test prediction
    prediction = fitter.predict(local=False)
    print(f"✓ Prediction successful, shape: {prediction.shape}")
    
    print("Testing model fit...")
    # Test a very short fit
    try:
        fitter.fit_global(maxiter=2, tol=1e-2, step_size=0.01)
        print("✓ Model fit successful!")
    except Exception as e:
        print(f"✗ Model fit failed: {e}")
        raise
    
    print("All tests passed!")
    
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()