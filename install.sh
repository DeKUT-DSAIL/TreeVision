#!/bin/bash

# Get the working directory
# WORKING_DIR=$(pwd)

# Requirements file
REQ_FILE=requirements.txt

source ~/miniconda3/etc/profile.d/conda.sh

# Install required packages
conda install kivy=2.1.0 -c conda-forge
pip install -r $REQ_FILE

# Run the application
python main.py