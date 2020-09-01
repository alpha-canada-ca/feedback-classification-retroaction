FROM dtobtn/python-ml-base:latest

COPY by_page.py /app/
COPY fetch_table.py /app/
COPY docker/feedback-by-page/entrypoint.sh /app/
COPY ./docker/train/crontab /crontab
RUN chmod a+x /app/fetch_table.py
COPY docker/feedback-by-page/entrypoint.sh /app/
RUN chmod a+x /app/entrypoint.sh
RUN crontab /crontab
COPY data/ /app/data/
RUN mkdir /app/config
COPY ./config/config.ini /app/config/config.ini
COPY ./templates/info_by_page_en.html /app/templates/info_by_page_en.html
COPY ./templates/info_by_page_en.html /app/templates/info_by_page_fr.html
WORKDIR /app

ENV FLASK_APP=by_page.py

ENTRYPOINT [ "/app/entrypoint.sh" ]


