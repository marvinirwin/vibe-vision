#!/bin/bash

# Navigate to the directory where the script is located
cd "$(dirname "$0")"

# Define the virtual environment directory name
VENV_DIR="venv"

# Check if the virtual environment directory exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found. Creating one..."
    # Create the virtual environment using python3
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment."
        exit 1
    fi
    echo "Virtual environment created successfully."
else
    echo "Virtual environment already exists."
fi

# Define the path to the Python executable within the venv
VENV_PYTHON="$VENV_DIR/bin/python"

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found in the current directory."
    exit 1
fi

# Install dependencies using pip from the virtual environment
echo "Installing dependencies from requirements.txt..."
"$VENV_PYTHON" -m pip install --upgrade pip # Upgrade pip first
"$VENV_PYTHON" -m pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies."
    exit 1
fi

echo "Setup complete. Virtual environment '$VENV_DIR' is ready."
echo "To activate it, run: source $VENV_DIR/bin/activate"

exit 0 