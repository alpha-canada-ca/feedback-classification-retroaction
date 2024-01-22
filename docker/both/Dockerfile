# Use a base image with Python
FROM python:3.8-slim

# Install cron and other system dependencies
RUN apt-get update && apt-get install -y cron

# Install Flask and your other Python dependencies
RUN pip install -U Flask gspread oauth2client pyairtable

# Copy and set permissions for your scripts and crontab file
COPY suggest.py /app/
COPY client_secret.json /app/
COPY process_feedback.py /app/
COPY docker/both/entrypoint.sh /app/
COPY ./docker/train/crontab /crontab
RUN chmod a+x /app/process_feedback.py
RUN chmod a+x /app/entrypoint.sh

# Apply crontab
RUN crontab /crontab

# Copy data and config files
COPY data/ /app/data/
COPY ./config/config.ini /app/config/

# Set the working directory and Flask environment variable
WORKDIR /app
ENV FLASK_APP=suggest.py

# Set the entrypoint for the container
ENTRYPOINT [ "/app/entrypoint.sh" ]
