FROM python:3.8-slim

# Install any additional system dependencies here if necessary

# Copy your application files
COPY suggest.py /app/
COPY client_secret.json /app/
COPY process_feedback.py /app/
COPY docker/both/entrypoint.sh /app/
COPY ./docker/train/crontab /crontab
COPY data/ /app/data/
COPY ./config/config.ini /app/config/

# Install your Python dependencies
RUN pip install -U gspread oauth2client pyairtable

# Set up file permissions and cron job
RUN chmod a+x /app/process_feedback.py
RUN chmod a+x /app/entrypoint.sh
RUN crontab /crontab

# Set the working directory and Flask environment variable
WORKDIR /app
ENV FLASK_APP=suggest.py

# Set the entrypoint for the container
ENTRYPOINT [ "/app/entrypoint.sh" ]
