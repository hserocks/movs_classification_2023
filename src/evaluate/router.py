import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi_cache.decorator import cache
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_async_session
from src.evaluate.models import evaluate_table, evaluate_fmodels
from src.evaluate.schemas import ModelEvaluate, FeatureModelEvaluate

from celery import Celery
import celery_aio_pool as aio_pool


import pytz
from dateutil import parser # python-dateutil

from train import main as train_main
from train_svm_xgb import main as train_svm_xgb

from src.config import REDIS_HOST, REDIS_PORT
import asyncio

# main(model = 'vit', image_folder_path = 'Data_small', eval = False):
    
router = APIRouter(
    prefix="/evaluate",
    tags=["Evaluate"]
)

celery = Celery('evaluate', 
                broker=f'redis://{REDIS_HOST}:{REDIS_PORT}',
                backend=f'redis://{REDIS_HOST}:{REDIS_PORT}', 
                worker_pool=aio_pool.pool.AsyncIOPool)


'''
@router.post("")
async def evaluate(new_operation: ModelEvaluate, session: AsyncSession = Depends(get_async_session)):
    from datetime import datetime
    timestamp = datetime.now().astimezone(pytz.utc).replace(tzinfo=None)
    
    stmt = insert(evaluate_table).values(
        **new_operation.model_dump(),
        date=timestamp
    )
    
    await session.execute(stmt)
    await session.commit()
    
    #output = await cached_evaluate(new_operation.model, new_operation.data, new_operation.evaluate_only)
    output = cached_evaluate.delay(new_operation.model, new_operation.data, new_operation.evaluate_only)
    
    return {"status": "success", "result": output}

@celery.task
def cached_evaluate(model: str, data: str, eval_only: bool):
    return cached_evaluate2(model, data, eval_only)

@cache(expire=900)
def cached_evaluate2(model: str, data: str, eval_only: bool):
    return train_main(model, data, eval_only)
'''

@router.get("/celery_test")
def test():
    test_task.delay()

@celery.task
def test_task():
    print("Hello from Celery!")
    return('Hello from Celery!')
    


@router.post("")
async def evaluate(new_operation: ModelEvaluate, session: AsyncSession = Depends(get_async_session)):
    from datetime import datetime
    timestamp = datetime.now().astimezone(pytz.utc).replace(tzinfo=None)
    
    stmt = insert(evaluate_table).values(
        **new_operation.model_dump(),
        date=timestamp
    )
    
    await session.execute(stmt)
    await session.commit()
    
    # Dispatch the background task without waiting for the result
    task = cached_evaluate.delay(new_operation.model, new_operation.data, new_operation.evaluate_only)
    #task = cached_evaluate.delay('a', 'b', 'c')
    
    return {"status": "success", "task_id": task.id}  # Return a task ID or similar identifier





@celery.task
def cached_evaluate(model: str, data: str, eval_only: bool):
    #print('wtf')
    #return 'wtf'
    return train_main(model, data, eval_only)


from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool


# @cache(expire=900)  # Cache based on the task ID
@router.get("/result/{task_id}")
async def get_task_result(task_id: str):
    task = cached_evaluate.AsyncResult(task_id)

    # Check if task is ready without blocking
    is_ready = task.ready
    if not is_ready:
        raise HTTPException(status_code=202, detail="Task is still processing")

    # Retrieve task result asynchronously using run_in_threadpool
    #result = await run_in_threadpool(task.get, timeout=10)  # Adjust the timeout as necessary
    #result = task.get
    try:
        result = await run_in_threadpool(task.get, timeout=5)
    except Exception as e:
        print(e)
        return {"status": "in progress", "result": "not ready yet"}
    return {"status": "success", "result": result}


# FMODELS


@router.post("/fmodels")
async def evaluate(new_operation: FeatureModelEvaluate, session: AsyncSession = Depends(get_async_session)):
    from datetime import datetime
    timestamp = datetime.now().astimezone(pytz.utc).replace(tzinfo=None)
    
    stmt = insert(evaluate_fmodels).values(
        **new_operation.model_dump(),
        date=timestamp
    )
    
    await session.execute(stmt)
    await session.commit()
    
    # Dispatch the background task without waiting for the result
    task = cached_evaluate_fmodels.delay(new_operation.features, new_operation.fmodel, new_operation.evaluate_only)
    #task = cached_evaluate.delay('a', 'b', 'c')
    
    return {"status": "success", "task_id": task.id}  # Return a task ID or similar identifier

@celery.task
def cached_evaluate_fmodels(features: str, fmodel: str, eval_only: bool):
    return train_svm_xgb(features, fmodel, eval_only)

# @cache(expire=900)
# def cached_evaluate_fmodel2(features: str, fmodel: str, eval_only: bool):
#     return train_svm_xgb(features, fmodel, eval_only)