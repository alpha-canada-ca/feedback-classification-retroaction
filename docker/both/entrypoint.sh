#!/bin/bash
echo "Running process_feedback.py"
python3 /app/process_feedback.py
echo "Starting Flask app"
flask run --host 0.0.0.0
