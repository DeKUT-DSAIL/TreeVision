#!/bin/bash

working_dir=$(pwd)

source "$working_dir"/venv/bin/activate

python "$working_dir"/main.py
