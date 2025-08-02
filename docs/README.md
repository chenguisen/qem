# QEM Documentation

This directory contains the documentation for the QEM (Quantitative Electron Microscopy) package.

## Building the Documentation

### Prerequisites

```bash
pip install sphinx sphinx_rtd_theme myst-parser
```

### Quick Build

```bash
# Using the build script (recommended)
python build_docs.py

# Or using make directly
make html
```

### Serving Locally

```bash
# Serve on http://localhost:8000
python serve_docs.py

# Or serve on a different port
python serve_docs.py 8080
```

## Documentation Structure

```
docs/
├── source/                     # Source files
│   ├── index.rst              # Main page
│   ├── installation.rst       # Installation guide
│   ├── quickstart.rst         # Quick start guide
│   ├── api/                   # API documentation
│   │   ├── index.rst
│   │   └── modules.rst
│   ├── tutorials/             # Tutorials
│   │   ├── index.rst
│   │   └── basic_analysis.rst
│   ├── user_guide/           # User guides
│   ├── development/          # Development docs
│   └── about/               # About pages
├── build/                    # Generated documentation
│   └── html/                # HTML output
├── build_docs.py            # Build script
├── serve_docs.py            # Local server script
└── README.md               # This file
```

## Key Features

- **Comprehensive API Documentation**: Auto-generated from docstrings
- **Step-by-Step Tutorials**: Learn QEM through practical examples
- **Installation Guides**: Get QEM running on your system
- **Developer Documentation**: Contribute to QEM development
- **Modern Theme**: Clean, responsive ReadTheDocs theme

## Writing Documentation

### Adding New Pages

1. Create `.rst` file in appropriate `source/` subdirectory
2. Add to relevant `toctree` directive in parent `index.rst`
3. Rebuild documentation

### Tutorial Template

```rst
Tutorial Title
==============

Brief description of what this tutorial covers.

Learning Objectives
-------------------

- Objective 1
- Objective 2

Prerequisites
-------------

- Requirement 1
- Requirement 2

Step 1: Description
-------------------

.. code-block:: python

   # Example code
   import qem
   
Text explaining the code...

Next Steps
----------

Links to related tutorials or advanced topics.
```

### API Documentation

API documentation is auto-generated from docstrings. Use NumPy-style docstrings:

```python
def example_function(param1: str, param2: int = 0) -> bool:
    """
    Brief description of the function.
    
    Parameters
    ----------
    param1 : str
        Description of param1
    param2 : int, default=0
        Description of param2
        
    Returns
    -------
    bool
        Description of return value
        
    Examples
    --------
    >>> example_function("test", 5)
    True
    """
    return True
```

## Troubleshooting

### Common Issues

**ImportError during build:**
- Missing dependencies are mocked in `conf.py`
- Warnings are expected for missing optional dependencies

**Broken links:**
- External links may fail during local builds
- Use `python build_docs.py --check-links` to verify

**Build fails:**
- Check Sphinx installation: `sphinx-build --version`
- Verify Python path in `conf.py`
- Check for syntax errors in `.rst` files

### Getting Help

- Check the [Sphinx documentation](https://www.sphinx-doc.org/)
- Review existing documentation files for examples
- Ask questions in GitHub discussions

## Automated Builds

For production deployment, consider setting up automated builds with:

- **GitHub Actions**: Build docs on every commit
- **ReadTheDocs**: Automatic hosting and building
- **GitHub Pages**: Host static documentation

Example GitHub Actions workflow:

```yaml
name: Build Documentation

on: [push, pull_request]

jobs:
  docs:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        pip install -e .
        pip install sphinx sphinx_rtd_theme myst-parser
    - name: Build documentation
      run: |
        cd docs
        python build_docs.py
```

## Contributing

Documentation contributions are welcome! Please:

1. Follow the existing style and structure
2. Test your changes locally before submitting
3. Include examples in tutorials
4. Update relevant sections when adding features

Thank you for helping improve QEM documentation! 📚