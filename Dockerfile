FROM python:3.11.8

RUN mkdir /fastapi_app

WORKDIR /fastapi_app

# clone git repo
# RUN git clone https://github.com/hserocks/movs_classification_2023.git .

COPY requirements.txt .
RUN pip install -r requirements.txt

# run script to download data
COPY downloader.py .
RUN python downloader.py --selection all

COPY . .

RUN chmod a+x docker/*.sh

# Install dockerize
# ENV DOCKERIZE_VERSION v0.6.1
# RUN wget https://github.com/jwilder/dockerize/releases/download/$DOCKERIZE_VERSION/dockerize-linux-amd64-$DOCKERIZE_VERSION.tar.gz \
#     && tar -C /usr/local/bin -xzvf dockerize-linux-amd64-$DOCKERIZE_VERSION.tar.gz \
#     && rm dockerize-linux-amd64-$DOCKERIZE_VERSION.tar.gz


# WORKDIR src

#CMD celery -A src.evaluate.router:celery worker --loglevel=INFO && gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind=0.0.0.0:8000
# CMD gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind=0.0.0.0:8000

# Start the bot and the FastAPI app
# CMD ["sh", "-c", "nohup python bot_new.py > /dev/null 2>&1 & exec gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind=0.0.0.0:8000"]

ENTRYPOINT ["/fastapi_app/start.sh"]