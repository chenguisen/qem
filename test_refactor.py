#!/usr/bin/env python3
"""Test script for the refactored ImageModelFitting class"""

import numpy as np
import sys
import os

def test_syntax_check():
    """Test that the refactored file has valid Python syntax"""
    try:
        import qem.model
        print("✓ qem.model imports successfully")
        
        from qem.model import GaussianModel, set_backend
        print("✓ Backend abstraction imports successful")
        
        set_backend('jax')
        print("✓ Backend setting works")
        
        return True
        
    except Exception as e:
        print(f"✗ Syntax/import test failed: {str(e)}")
        return False

def test_image_fitting_syntax():
    """Test ImageModelFitting syntax without running complex operations"""
    try:
        import qem.image_fitting
        print("✓ qem.image_fitting imports successfully")
        
        from qem.image_fitting import ImageModelFitting
        print("✓ ImageModelFitting class imports successfully")
        
        print("✓ Refactored ImageModelFitting has valid syntax")
        return True
        
    except SyntaxError as e:
        print(f"✗ Syntax error in ImageModelFitting: {str(e)}")
        return False
    except Exception as e:
        print(f"✗ Import error in ImageModelFitting: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing refactored ImageModelFitting class syntax...")
    print("=" * 50)
    
    success = True
    success &= test_syntax_check()
    success &= test_image_fitting_syntax()
    
    print("=" * 50)
    if success:
        print("✓ Syntax tests passed! Refactoring appears successful.")
        print("Note: Full functionality testing requires additional dependencies.")
    else:
        print("✗ Syntax tests failed!")
        sys.exit(1)
