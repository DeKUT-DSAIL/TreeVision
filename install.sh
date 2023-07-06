#!/bin/bash

# Get the working directory
WORKING_DIR = $(pwd)

# Software Version
VERSION = 0.1.0

# Define the name of the virtual environment
VENV_NAME = "treevision"

# Requirements file
REQ_FILE = "$WORKING_DIR"/requirements.txt

# Software name
SOFTWARE_NAME = "TreeVision"

# Create virtual environment
conda create -n $VENV_NAME python=3.9

# Activate virtual environment
conda activate $VENV_NAME

# Install required packages
conda install kivy=2.1.0 -c conda-forge
pip install -r $REQ_FILE

# Run the application
python main.py