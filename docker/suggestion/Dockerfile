FROM ubuntu:18.04

# Install Python and basic utilities
RUN yes | apt-get update \
    && yes | apt-get upgrade \
    && yes | apt-get install python3 python3-pip

# Set the locale
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y locales \
    && sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && dpkg-reconfigure --frontend=noninteractive locales \
    && update-locale LANG=en_US.UTF-8
ENV LANG en_US.UTF-8 

# Install Python packages
RUN pip3 install -U Flask \
    nltk \
    requests \
    pandas \
    gspread \
    oauth2client \
    scikit-learn \
    numpy \
    pyairtable

# Download NLTK data
RUN python3 -m nltk.downloader stopwords

# Create app directory
RUN mkdir /app

# Copy application files
COPY suggest.py /app/
COPY /data /app/data

# Set working directory
WORKDIR /app

# Set Flask environment
ENV FLASK_APP=suggest.py

# Start Flask application
ENTRYPOINT [ "flask", "run", "--host", "0.0.0.0" ]
