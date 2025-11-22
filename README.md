# Gesture Controlled Snake Game

A gesture-controlled Snake game using MediaPipe (hand tracking), OpenCV and NumPy.

## Setup

1. Create a Python 3.11 virtual environment (recommended):

```bash
cd "/Users/aryanrajput/Downloads/snake game"
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

2. Run the game:

```bash
source .venv/bin/activate
python main.py
# Or use the provided wrapper:
./run.sh
```

## Files added
- `requirements.txt` — pinned dependency versions.
- `.gitignore` — ignores `.venv`, caches, and editor folders.
- `run.sh` — small starter script to activate venv and run the game.
- `.vscode/settings.json` — (already present) points VS Code to the `.venv`.

## Prepare and push to GitHub
```bash
# Initialize git (if not already initialized)
git init
git add .
git commit -m "Initial project files: gesture-controlled snake game"
# Create a repository on GitHub, then add the remote and push
git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME/YOUR_REPO` with your repository information.

## Notes
- The `.venv` directory is intentionally ignored; do not commit it.
- If you prefer conda/mambaforge for binary dependencies on macOS, I can provide an `environment.yml`.

Have fun — let me know if you want me to initialize the git repo and make the initial commit for you.
