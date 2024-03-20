import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi_cache.decorator import cache
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_async_session
from src.operations.models import operation
from src.operations.schemas import OperationCreate

router = APIRouter(
    prefix="/operations",
    tags=["Operation"]
)




