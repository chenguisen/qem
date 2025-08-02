#!/usr/bin/env python3
"""
Build script for QEM documentation
"""
import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, check=True, 
                              capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def build_docs():
    """Build the documentation"""
    docs_dir = Path(__file__).parent
    
    print("🔧 Building QEM Documentation")
    print("=" * 50)
    
    # Check if sphinx is installed
    success, _ = run_command("sphinx-build --version")
    if not success:
        print("❌ Sphinx not found. Installing...")
        success, output = run_command("pip install sphinx sphinx_rtd_theme myst-parser")
        if not success:
            print(f"❌ Failed to install Sphinx: {output}")
            return False
    
    # Clean previous build
    print("🧹 Cleaning previous build...")
    success, _ = run_command("make clean", cwd=docs_dir)
    if not success:
        print("⚠️  Clean failed, continuing...")
    
    # Build HTML documentation
    print("📚 Building HTML documentation...")
    success, output = run_command("make html", cwd=docs_dir)
    
    if success:
        print("✅ Documentation built successfully!")
        print(f"📂 Output directory: {docs_dir / 'build' / 'html'}")
        
        # Check if index.html exists
        index_file = docs_dir / "build" / "html" / "index.html"
        if index_file.exists():
            print(f"🌐 Main page: file://{index_file}")
            print("💡 To serve locally: python serve_docs.py")
        else:
            print("⚠️  Index file not found")
            
        return True
    else:
        print("❌ Documentation build failed!")
        print(f"Error output: {output}")
        return False

def check_links():
    """Check for broken links in documentation"""
    docs_dir = Path(__file__).parent
    
    print("\n🔗 Checking for broken links...")
    success, output = run_command("make linkcheck", cwd=docs_dir)
    
    if success:
        print("✅ Link check completed")
    else:
        print("⚠️  Link check had issues (this is normal for local builds)")

if __name__ == "__main__":
    success = build_docs()
    
    if success and "--check-links" in sys.argv:
        check_links()
    
    sys.exit(0 if success else 1)