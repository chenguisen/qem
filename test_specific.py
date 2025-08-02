#!/usr/bin/env python3
"""Test the specific functionality that was failing."""

try:
    print("Testing specific import that was failing...")
    
    # This is the import that was failing
    from qem.io import read_legacyInputStatSTEM
    print("✓ Successfully imported read_legacyInputStatSTEM")
    
    # Test other imports from benchmark.py
    from qem.image_fitting import ImageFitting
    print("✓ Successfully imported ImageFitting")
    
    from qem.utils import safe_convert_to_numpy
    print("✓ Successfully imported safe_convert_to_numpy")
    
    print("All specific imports successful!")
    
except Exception as e:
    print(f"✗ Specific import failed: {e}")
    import traceback
    traceback.print_exc()