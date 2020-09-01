FROM dtobtn/python-ml-base:1.0.0

RUN python3 -mpip install matplotlib
COPY by_page.py /app/
COPY fetch_table.py /app/
COPY docker/feedback_by_page/entrypoint.sh /app/
COPY ./docker/train/crontab /crontab
RUN chmod a+x /app/fetch_table.py
COPY docker/feedback_by_page/entrypoint.sh /app/
RUN chmod a+x /app/entrypoint.sh
RUN crontab /crontab
COPY data/ /app/data/
RUN mkdir /app/config
COPY ./config/config.ini /app/config/config.ini
WORKDIR /app

ENV FLASK_APP=by_page.py

ENTRYPOINT [ "/app/entrypoint.sh" ]


