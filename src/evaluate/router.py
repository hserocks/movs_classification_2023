import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi_cache.decorator import cache
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_async_session
from src.evaluate.models import evaluate_table, evaluate_fmodels
from src.evaluate.schemas import ModelEvaluate, FeatureModelEvaluate



import pytz
from dateutil import parser # python-dateutil

from train import main as train_main
from train_svm_xgb import main as train_svm_xgb

# main(model = 'vit', image_folder_path = 'Data_small', eval = False):
    
router = APIRouter(
    prefix="/evaluate",
    tags=["Evaluate"]
)
    
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
    
    output = await cached_evaluate(new_operation.model, new_operation.data, new_operation.evaluate_only)
    
    return {"status": "success", "result": output}


@cache(expire=900)
async def cached_evaluate(model: str, data: str, eval_only: bool):
    return train_main(model, data, eval_only)



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
    
    output = await cached_evaluate_fmodels(new_operation.features, new_operation.fmodel, new_operation.evaluate_only)
    
    return {"status": "success", "result": output}


@cache(expire=900)
async def cached_evaluate_fmodels(features: str, fmodel: str, eval_only: bool):
    return train_svm_xgb(features, fmodel, eval_only)