#!/usr/bin/env bash

#sudo apt update
#sudo apt -y install python3.10-venv nvtop

rm venv -rf

# Create and activate virtual environment
python3.11 -m venv --without-pip venv
source venv/bin/activate

# Install all required packages
pip install -r requirements.txt

# Instructions for user
echo ""
echo "To activate this environment in the future, only run:"
echo "source venv/bin/activate"
echo ""
