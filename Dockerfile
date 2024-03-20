FROM python:3.11.8

RUN mkdir /fastapi_app

WORKDIR /fastapi_app

# clone git repo
RUN git clone https://github.com/hserocks/movs_classification_2023.git .

COPY requirements.txt .

RUN pip install -r requirements.txt

# run script to download data
RUN python downloader.py --selection all

COPY . .

RUN chmod a+x docker/*.sh

# WORKDIR src

CMD celery -A src.evaluate.router:celery worker --loglevel=INFO && gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind=0.0.0.0:8000