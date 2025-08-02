#!/usr/bin/env python3
"""Simple script to test imports."""

try:
    print("Testing qem.io import...")
    import qem.io
    print("✓ qem.io imported successfully")
    
    print("Testing qem.utils import...")
    import qem.utils
    print("✓ qem.utils imported successfully")
    
    print("Testing qem.backend_utils import...")
    import qem.backend_utils
    print("✓ qem.backend_utils imported successfully")
    
    print("Testing backend detection...")
    available = qem.backend_utils.detect_available_backends()
    print(f"✓ Available backends: {available}")
    
    if available:
        print("Testing backend setup...")
        backend = qem.backend_utils.setup_test_backend()
        print(f"✓ Selected backend: {backend}")
    
    print("All imports successful!")
    
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()