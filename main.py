from sqlalchemy import create_engine, text
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from redis import asyncio as aioredis
from src.auth.base_config import auth_backend, fastapi_users
from src.auth.schemas import UserCreate, UserRead
from src.config import REDIS_HOST, REDIS_PORT
from src.operations.router import router as router_operation
from src.auth.router import router as role_adding_router
from src.pages.router import router as router_pages
from src.inference.router import router as router_inference  # NEW!!!
from src.evaluate.router import router as router_evaluate  # NEW!!!

# TEMP NEW
from src.config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASS

# add the parent directory to the sys path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# TEMP NEW
# from docker.postgres_test import create_conn_main
# import asyncio
# engine = asyncio.run(create_conn_main())
# TEMP NEW

engine = create_engine(f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
sql = text("INSERT INTO role (id, name, permissions) \
            VALUES (1, 0, NULL) ON CONFLICT (id) DO NOTHING;")
with engine.connect() as conn:
    conn.execute(sql)
    conn.commit()  # Make sure to commit the transaction
# TEMP NEW


app = FastAPI(
    title="Image classifier!"
)

app.mount("/static", StaticFiles(directory="src/static"), name="static")

app.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/auth",
    tags=["Auth"],
)

app.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/auth",
    tags=["Auth"],
)

app.include_router(router_operation)
app.include_router(router_inference)  # NEW!!!
app.include_router(router_evaluate)  # NEW!!!


app.include_router(router_pages)
app.include_router(role_adding_router)


origins = [
    "http://localhost:3000", "http://178.128.171.95:3000",
    "http://178.128.171.95:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "DELETE", "PATCH", "PUT"],
    allow_headers=[
        "Content-Type",
        "Set-Cookie",
        "Access-Control-Allow-Headers",
        "Access-Control-Allow-Origin",
        "Authorization"],
)


@app.on_event("startup")
async def startup_event():
    redis = aioredis.from_url(f"redis://{REDIS_HOST}:{REDIS_PORT}", encoding="utf8", decode_responses=True)
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")
