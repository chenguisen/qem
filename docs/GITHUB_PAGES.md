# GitHub Pages Documentation

This repository uses GitHub Actions to automatically build and deploy documentation to GitHub Pages.

## Automatic Deployment

Documentation is automatically built and deployed when:
- Code is pushed to the `master` branch
- Pull requests are created (build only, no deployment)

## Manual Setup Required

To enable GitHub Pages for this repository, you need to:

1. **Go to your GitHub repository settings**
2. **Navigate to Pages section** (Settings → Pages)
3. **Set Source to "GitHub Actions"**
4. **Save the configuration**

## Workflow Details

The documentation workflow (`.github/workflows/docs.yml`):
- Builds documentation using Sphinx
- Uses Python 3.9 and installs required dependencies
- Deploys to GitHub Pages on successful builds from master branch
- Supports both `requirements.txt` and `pyproject.toml` dependency management

## Accessing Documentation

Once set up, your documentation will be available at:
```
https://[your-username].github.io/[repository-name]/
```

## Local Development

To build documentation locally:
```bash
cd docs
python build_docs.py
```

To serve documentation locally:
```bash
cd docs
python serve_docs.py
```