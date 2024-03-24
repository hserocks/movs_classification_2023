#!/bin/bash

#cd src

if [[ "${1}" == "celery" ]]; then
  celery --app=src.evaluate.router:celery worker -l INFO
elif [[ "${1}" == "flower" ]]; then
  celery --app=src.evaluate.router:celery flower
 fi