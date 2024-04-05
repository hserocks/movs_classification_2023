import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.exc import OperationalError

from src.config import DB_HOST, DB_NAME, DB_PASS, DB_PORT, DB_USER

DATABASE_URL = \
    f"postgresql+asyncpg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


async def create_conn():
    engine = None
    while engine is None:
        try:
            # Format: "postgresql+asyncpg://<user>:<password>@<host>/<dbname>"
            database_url = DATABASE_URL
            engine = create_async_engine(database_url)

            # Async engine does not have .connect(),
            # instead use .begin() for transactions
            async with engine.begin() as conn:
                # Perform a simple query to check the connection
                await conn.execute("SELECT 1")
                print("Database connection successful")
        except OperationalError as e:
            print(e)
            await asyncio.sleep(5)  # Use asyncio.sleep for async context
    return engine


# Running the async function
async def create_conn_main():
    engine = await create_conn()

if __name__ == "__main__":
    asyncio.run(create_conn_main())
