import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytz

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
# from fastapi_cache.decorator import cache


from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert
# from sqlalchemy import insert, select

from src.database import get_async_session
from src.evaluate.models import evaluate_table, evaluate_fmodels
from src.evaluate.schemas import ModelEvaluate, FeatureModelEvaluate
from src.config import REDIS_HOST, REDIS_PORT

from celery import Celery
import celery_aio_pool as aio_pool


from scripts.train import main as train_main
from scripts.train_svm_xgb import main as train_svm_xgb

# import time
# import asyncio
# from dateutil import parser # python-dateutil

# main(model = 'vit', image_folder_path = 'Data_small', eval = False):

router = APIRouter(
    prefix="/evaluate",
    tags=["Evaluate"]
)

celery = Celery('evaluate',
                broker=f'redis://{REDIS_HOST}:{REDIS_PORT}',
                backend=f'redis://{REDIS_HOST}:{REDIS_PORT}',
                worker_pool=aio_pool.pool.AsyncIOPool)

# TEMP FOR TESTING
# celery = Celery('evaluate',
#                 broker=f'redis://{REDIS_HOST}:{REDIS_PORT}',
#                 backend=f'redis://{REDIS_HOST}:{REDIS_PORT}',
#                 )
# TEMP FOR TESTING


@router.post("")
async def evaluate(new_operation: ModelEvaluate,
                   session: AsyncSession = Depends(get_async_session)):
    from datetime import datetime
    timestamp = datetime.now().astimezone(pytz.utc).replace(tzinfo=None)

    stmt = insert(evaluate_table).values(
        **new_operation.model_dump(),
        date=timestamp
    )

    await session.execute(stmt)
    await session.commit()

    # Dispatch the background task without waiting for the result
    task = cached_evaluate.delay(new_operation.model,
                                 new_operation.data,
                                 new_operation.evaluate_only)
    # task = cached_evaluate.delay('a', 'b', 'c')

    # Return a task ID or similar identifier
    return {"status": "success", "task_id": task.id}


@celery.task
def cached_evaluate(model: str, data: str, eval_only: bool):
    return train_main(model, data, eval_only)


# @cache(expire=900)  # Cache based on the task ID
@router.get("/result/{task_id}")
async def get_task_result(task_id: str):
    task = cached_evaluate.AsyncResult(task_id)

    # Check if task is ready without blocking
    is_ready = task.ready
    if not is_ready:
        raise HTTPException(status_code=202, detail="Task is still processing")

    try:
        result = await run_in_threadpool(task.get, timeout=5)
    except Exception as e:
        print(e)
        return {"status": "in progress", "result": "not ready yet"}
    return {"status": "success", "result": result}

# FMODELS


@router.post("/fmodels")
async def evaluate(new_operation: FeatureModelEvaluate,
                   session: AsyncSession = Depends(get_async_session)):
    from datetime import datetime
    timestamp = datetime.now().astimezone(pytz.utc).replace(tzinfo=None)

    stmt = insert(evaluate_fmodels).values(
        **new_operation.model_dump(),
        date=timestamp
    )

    await session.execute(stmt)
    await session.commit()

    # Dispatch the background task without waiting for the result
    task = cached_evaluate_fmodels.delay(new_operation.features,
                                         new_operation.fmodel,
                                         new_operation.evaluate_only)

    # Return a task ID or similar identifier
    return {"status": "success", "task_id": task.id}


@celery.task
def cached_evaluate_fmodels(features: str, fmodel: str, eval_only: bool):
    return train_svm_xgb(features, fmodel, eval_only)
