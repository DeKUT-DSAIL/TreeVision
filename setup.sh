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
eval "$(conda shell.bash hook)"
conda activate $VENV_NAME

REQ_FILE=requirements.txt

# Install required packages
conda install kivy=2.1.0 -c conda-forge
pip install -r $REQ_FILE
