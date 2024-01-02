#!/bin/zsh

# Get the current working directory, which is assumed to be the project root
PROJECT_ROOT=$(pwd)

# The utils directory path
UTILS_PATH="$PROJECT_ROOT/utils"

# Check if .zshrc already contains the path
if grep -q "$UTILS_PATH" ~/.zshrc; then
    echo "Utils path already in PYTHONPATH"
else
    # Append the utils path to PYTHONPATH in .zshrc
    echo "export PYTHONPATH=\"\$PYTHONPATH:$UTILS_PATH\"" >> ~/.zshrc
    echo "Added $UTILS_PATH to PYTHONPATH in .zshrc"
fi
