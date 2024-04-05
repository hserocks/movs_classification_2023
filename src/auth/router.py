from fastapi import APIRouter, Depends
from sqlalchemy import insert
from src.auth.models import role
from src.auth.schemas import RoleCreate
from sqlalchemy.ext.asyncio import AsyncSession
from src.database import get_async_session

router = APIRouter(
    prefix="/roles",
    tags=["Operation"]
)


@router.post("/add")
async def add_role(new_operation: RoleCreate,
                   session: AsyncSession = Depends(get_async_session)):
    stmt = insert(role).values(**new_operation.dict())
    await session.execute(stmt)
    await session.commit()
    return {"status": "success"}
