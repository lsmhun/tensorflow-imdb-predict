#!/bin/bash
echo 'Starting IMDb comment predict application'
source venv/bin/activate
python3 lsmhun/train_model.py
