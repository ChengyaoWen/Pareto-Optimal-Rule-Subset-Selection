#!/bin/bash
pip install -r requirements.txt

# unzip data
cd datasets || exit
bash ./generate_data.sh
cd ../

# run all experiments
cd pors || exit
python experiment.py
cd ../
