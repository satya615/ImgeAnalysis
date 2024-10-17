#!/bin/bash

echo " BUILD START"

# Use 'python3' instead of 'python3.10' since Vercel might not have Python 3.10 installed by default
python3 -m pip install -r requirements.txt
python3 manage.py collectstatic --noinput --clear

# Ensure the output directory exists and is correctly handled
mkdir -p staticfiles_build
mv static/* staticfiles_build/

echo " BUILD END"
