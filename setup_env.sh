#!/bin/bash

# Script to create virtual environment and install requirements

set -e  # Exit on error

VENV_DIR="venv"
REQUIREMENTS_FILE="requirements.txt"

echo "Creating virtual environment in '$VENV_DIR'..."
python3 -m venv $VENV_DIR

echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing packages from $REQUIREMENTS_FILE..."
pip install -r $REQUIREMENTS_FILE

echo ""
echo "Setup complete! Virtual environment created and packages installed."
echo "To activate the environment in the future, run:"
echo "  source $VENV_DIR/bin/activate"
