FROM dtobtn/python-ml-base:latest

COPY by_group.py /app/
COPY by_page.py /app/
COPY fetch_table.py /app/
COPY update_mongo_tags.py /app/
COPY app.py /app/
COPY index.py /app/
COPY group_index.py /app/
COPY page_index.py /app/
COPY docker/feedback-by-page/entrypoint.sh /app/
COPY ./docker/feedback-by-page/crontab /crontab
RUN chmod a+x /app/fetch_table.py
RUN chmod a+x /app/update_mongo_tags.py
COPY docker/feedback-by-page/entrypoint.sh /app/
RUN chmod a+x /app/entrypoint.sh
RUN crontab /crontab
COPY data/ /app/data/
RUN mkdir /app/config
COPY ./config/config.ini /app/config/config.ini
COPY templates/ /app/templates/
RUN pip3 install -U pip
RUN pip3 install -U pymongo
RUN pip3 install -U wordcloud
RUN pip3 install -U gspread
RUN pip3 install -U oauth2client

WORKDIR /app

ENV FLASK_APP=app.py

ENTRYPOINT [ "/app/entrypoint.sh" ]
