#!/bin/bash
cron
python /app/process_feedback.py &  # Run the script in the background
flask run --host 0.0.0.0
