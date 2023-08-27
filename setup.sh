#!/bin/bash

# Software name
# SOFTWARE_NAME="TreeVision"

# Software Version
# VERSION=0.1.0

# Define the name of the virtual environment
VENV_NAME="treevision"

# Create virtual environment
conda create -n $VENV_NAME python=3.9

# Activate virtual environment
source activate $VENV_NAME