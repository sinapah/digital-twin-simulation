#!/bin/bash
# =========================================================
# V4 Setup Script
# =========================================================
# Sets up the V4 environment on a VM
# Assumes digital-twin-simulation folder with UA-DETRAC data is mounted
# =========================================================

set -e

echo "=== V4 Setup ==="

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "=== Verification ==="
python3 -c "import torch; print('PyTorch:', torch.__version__)"
python3 -c "import torchvision; print('TorchVision:', torchvision.__version__)"
python3 -c "import pandas; print('Pandas:', pandas.__version__)"
python3 -c "import PIL; print('Pillow:', PIL.__version__)"
python3 -c "import psutil; print('PsUtil available')"

echo ""
echo "=== Setup Complete ==="
echo "Project directory: $SCRIPT_DIR"
echo ""
echo "To use:"
echo "  cd $SCRIPT_DIR"
echo "  source venv/bin/activate"
echo ""
