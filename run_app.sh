#!/bin/bash
# run_app.sh
# Starts the Pokémon Card Appraisal & Similarity Web App (FastAPI + React)
cd "$(dirname "$0")"
echo "Starting Pokémon Card Maturation Web App..."
echo "Open your browser to: http://localhost:8000"
python3 src/api.py
