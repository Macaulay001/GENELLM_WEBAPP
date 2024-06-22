#!/bin/bash

# Install system dependencies
apt-get update && apt-get install -y build-essential libssl-dev libffi-dev python3-dev

# Install Python dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt