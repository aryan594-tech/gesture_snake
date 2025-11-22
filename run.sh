#!/usr/bin/env bash
# Simple wrapper to activate the venv and run the game
set -e
cd "$(dirname "$0")"
if [ ! -d ".venv" ]; then
  echo "Virtual environment not found. Create one with: python3.11 -m venv .venv"
  exit 1
fi
source .venv/bin/activate
python main.py
