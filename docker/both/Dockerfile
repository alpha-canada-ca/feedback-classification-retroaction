FROM dtobtn/python-ml-base:latest

COPY suggest.py /app/
COPY process_feedback.py /app/
COPY docker/both/entrypoint.sh /app/
COPY ./docker/train/crontab /crontab
RUN pip3 install -U gspread
RUN pip3 install -U oauth2client
RUN chmod a+x /app/process_feedback.py
COPY docker/both/entrypoint.sh /app/
RUN chmod a+x /app/entrypoint.sh
RUN crontab /crontab
COPY data/ /app/data/
RUN mkdir /app/config
COPY ./config/config.ini /app/config/config.ini
WORKDIR /app

ENV FLASK_APP=suggest.py

ENTRYPOINT [ "/app/entrypoint.sh" ]


