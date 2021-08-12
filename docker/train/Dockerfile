FROM ubuntu:18.04
RUN yes | apt-get update
RUN yes | apt-get upgrade
RUN mkdir /app

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y locales

RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8

ENV LANG en_US.UTF-8 

RUN yes | apt-get install python3
RUN yes | apt-get install python3-pip
RUN yes | apt-get install cron

RUN pip3 install -U pandas
RUN pip3 install -U nltk
RUN pip3 install -U requests
RUN pip3 install -U scikit-learn
RUN pip3 install -U gspread
RUN pip3 install airtable-python-wrapper

COPY process_feedback.py /app/
RUN mkdir /app/data   
COPY ./docker/train/crontab /crontab
RUN chmod a+x /app/process_feedback.py


RUN crontab /crontab
ENTRYPOINT cron -f

WORKDIR /app
