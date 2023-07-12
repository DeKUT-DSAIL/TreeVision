#!/bin/bash

# Get the working dir
WORKING_DIR=$(pwd)

# Version of the software
VERSION=1.0.2

# Define the name of the virtual environment
VENV_NAME="fusion"

# Define the path to the requirements.txt file
REQ_FILE="$WORKING_DIR"/requirements.txt

# Define the name of the software and the command to start it
SOFTWARE_NAME="Fusion"

# Create the virtual environment
python3 -m venv $VENV_NAME

# Activate the virtual environment
source $VENV_NAME/bin/activate

# Install the required packages
pip install -r $REQ_FILE

# Create a desktop shortcut for the software
cat > ~/Desktop/$SOFTWARE_NAME.desktop <<EOF
[Desktop Entry]
Version=$VERSION
Exec=bash "$WORKING_DIR"/starter.sh
Icon="$WORKING_DIR"/assets/images/icons/forest_project.ico
Name=$SOFTWARE_NAME
GenericName=$SOFTWARE_NAME
Comment=Run the Fusion Forest Project Application
Encoding=UTF-8
Terminal=false
Type=Application
Catergories=Application;
EOF

# Make the shortcut executable
chmod +x ~/Desktop/$SOFTWARE_NAME.desktop
