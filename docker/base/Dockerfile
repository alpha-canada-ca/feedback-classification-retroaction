# Use a newer base image (Ubuntu 20.04) and reduce the number of RUN commands.
FROM ubuntu:20.04

# Update the package index and install system dependencies.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    locales \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cron

# Configure locales for UTF-8 support.
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8

# Set the environment variable for UTF-8 locale.
ENV LANG en_US.UTF-8

# Install Python packages including Flask and its dependencies.
RUN pip3 install -U Flask MarkupSafe nltk requests pandas gspread oauth2client datetime pymongo scikit-learn airtable-python-wrapper matplotlib gensim

# Create a directory for your application.
RUN mkdir /app

# Set the working directory to /app.
WORKDIR /app
