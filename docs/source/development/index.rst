Development Guide
=================

Welcome to the QEM development documentation. This guide is for contributors who want to help improve QEM.

.. toctree::
   :maxdepth: 2

   contributing
   architecture
   testing
   documentation
   release_process

Getting Started
---------------

To contribute to QEM:

1. Fork the repository on GitHub
2. Set up your development environment
3. Make your changes
4. Run tests and ensure code quality
5. Submit a pull request

Development Setup
-----------------

.. code-block:: bash

   # Clone your fork
   git clone https://github.com/yourusername/qem.git
   cd qem
   
   # Create development environment
   conda create -n qem-dev python=3.11
   conda activate qem-dev
   
   # Install in development mode
   pip install -e ".[dev]"
   
   # Install pre-commit hooks
   pre-commit install

Code Standards
--------------

QEM follows these coding standards:

- **PEP 8** style guide
- **Type hints** for all public functions
- **Docstrings** in NumPy format
- **Tests** for all new functionality
- **Ruff** for linting and formatting

Running Tests
-------------

.. code-block:: bash

   # Run all tests
   pytest
   
   # Run with coverage
   pytest --cov=qem
   
   # Run specific test file
   pytest tests/test_model.py

Building Documentation
----------------------

.. code-block:: bash

   cd docs
   make html
   
   # View documentation
   open build/html/index.html

Contributing Guidelines
-----------------------

Please see :doc:`contributing` for detailed guidelines on:

- Code review process
- Submitting issues
- Writing documentation
- Adding new features