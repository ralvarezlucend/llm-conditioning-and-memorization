#!/bin/bash

source .venv/bin/activate

# List available experiments if no argument provided
if [ $# -eq 0 ]; then
    echo "Available experiments:"
    ls conf/experiment/*.yaml | xargs -n 1 basename | sed 's/\.yaml$//' | sed 's/^/  - /'
    echo ""
    echo "Usage: $0 <experiment_name>"
    exit 0
fi

# Run the experiment
python main.py --config-name=$1 --multirun
