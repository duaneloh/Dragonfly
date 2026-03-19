#!/bin/bash
# Rebuild Sphinx documentation for Dragonfly

set -e

# Rebuild the Python extension (picks up any docstring changes)
cd ..
pip install -e . --no-build-isolation
cd docs

# Rebuild the docs
rm -rf _build
sphinx-build -b html . _build

echo "Docs built successfully. Serve with: python -m http.server 8000"
