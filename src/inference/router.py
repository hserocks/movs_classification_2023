import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi_cache.decorator import cache
from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_async_session
from src.inference.models import inference_table
from src.inference.schemas import InferenceCreate, InferenceGimages

import sys
from pathlib import Path
import os

import pytz
from dateutil import parser # python-dateutil

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
import shutil
import aiofiles

from typing import List
from sqlalchemy.orm import selectinload
import fastapi_users
from src.auth.models import User
from src.auth.base_config import current_user


# Add the original folder to sys.path
# curr_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(curr_dir)
#parent_dir = os.path.dirname(parent_dir)

# original_folder = Path(__file__).resolve().parents[1]
# sys.path.append(str(parent_dir))
# print('original_folder:', original_folder)  # temp

from inference import main as inference_main
from utils.gimages_dl import download_gimages, get_random_gimage
from random import choice
from shutil import copyfile

'''
import importlib
parent_dir = str(Path(__file__).resolve().parents[2])
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

module_name = "inference"  # Change to the name of your module
module = importlib.import_module(module_name)
inference_main = getattr(module, 'main', None)
'''


animal_list = ['Bear', 'Brown bear', 'Bull', 'Camel', 'Canary', 'Cat',
               'Caterpillar',
               'Cattle',
               'Centipede', 'Cheetah', 'Chicken', 'Crab',
               'Crocodile', 'Deer', 'Dog', 'Duck',
               'Eagle', 'Elephant', 'Fish', 'Fox', 'Frog',
               'Giraffe', 'Goat', 'Goldfish', 'Goose',
               'Hamster', 'Harbor seal', 'Hedgehog',
               'Hippopotamus', 'Horse', 'Jaguar', 'Jellyfish',
               'Kangaroo', 'Koala', 'Ladybug', 'Leopard',
               'Lion', 'Lizard', 'Lynx', 'Magpie',
               'Monkey', 'Moths and butterflies', 'Mouse',
               'Mule', 'Ostrich', 'Otter', 'Owl', 'Panda',
               'Parrot', 'Penguin', 'Pig', 'Polar bear',
               'Rabbit', 'Raccoon', 'Raven', 'Red panda',
               'Rhinoceros', 'Scorpion', 'Sea lion',
               'Sea turtle', 'Seahorse', 'Shark', 'Sheep',
               'Shrimp', 'Snail', 'Snake', 'Sparrow',
               'Spider', 'Squid', 'Squirrel', 'Starfish',
               'Swan', 'Tick', 'Tiger', 'Tortoise',
               'Turkey', 'Turtle', 'Whale', 'Woodpecker',
               'Worm', 'Zebra']



router = APIRouter(
    prefix="/inference",
    tags=["Inference"]
)

#current_user = fastapi_users.current_user()

@router.get("/long_operation")
@cache(expire=30)
def get_long_op():
    time.sleep(2)
    return "Много много данных, которые вычислялись сто лет"




@cache(expire=30)
async def cached_inference(model: str, link: str):
    return inference_main(model, link)

@router.post("")
async def add_inference(new_operation: InferenceCreate, session: AsyncSession = Depends(get_async_session), user: User = Depends(current_user)):
    from datetime import datetime
    timestamp = datetime.now().astimezone(pytz.utc).replace(tzinfo=None)
    
    stmt = insert(inference_table).values(
        **new_operation.model_dump(),
        date=timestamp
    )
    
    result = await session.execute(stmt)
    await session.commit()
    
    new_id = result.inserted_primary_key[0]
    # print('new_id:', new_id)  # temp
    output = await cached_inference(new_operation.model, new_operation.link)
    
    output_simple = None
    
    # Check if the output is a single word or a detailed result
    if '\n' in output:  # Detailed result
        # Split the output into lines and extract the first category after "Probability"
        lines = output.split('\n')
        for line in lines:
            if "Probability" in line:
                # The first category is in the line following "Probability"
                next_line_index = lines.index(line) + 1  # Get the index of the next line
                if next_line_index < len(lines):  # Check if the next line exists
                    output_simple = lines[next_line_index].strip().split()[0]  # Get the first word
                break  # No need to look further
    else:
        # Output is a single word
        output_simple = output
    
    update_stmt = update(inference_table).where(inference_table.c.id == new_id).values(output=output_simple)
    await session.execute(update_stmt)
    await session.commit()
    
    return {"status": "success", "inference": output}




@router.post("/gimages")
async def add_inference_gimages(new_operation: InferenceGimages, session: AsyncSession = Depends(get_async_session), user: User = Depends(current_user)):
    from datetime import datetime
    timestamp = datetime.now().astimezone(pytz.utc).replace(tzinfo=None)
    
    stmt = insert(inference_table).values(
         model = new_operation.model,
         link = f"{new_operation.query} (GImages)",
         date=timestamp
    )
    
    result = await session.execute(stmt)
    await session.commit()
    
    new_id = result.inserted_primary_key[0]
    # print('new_id:', new_id)  # temp
    if new_operation.query:
        image = download_gimages(new_operation.query)
        output = inference_main(new_operation.model, image)
    else:
        query = choice(animal_list)
        image = download_gimages(query)
        output = inference_main(new_operation.model, image)
        
    
    unique_identifier = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_filename = f"{unique_identifier}_{os.path.basename(image)}"
    static_file_dir = os.path.join('static', 'images')
    static_file_path = os.path.join(static_file_dir, new_filename)
    static_file_path_to_save = os.path.join('src', static_file_dir, new_filename)

    # Ensure the target directory exists
    os.makedirs(static_file_dir, exist_ok=True)

    # Copy the file from its original location to the static directory
    copyfile(image, static_file_path_to_save)
           
    # static_file_path = os.path.join('static', 'images', image.filename)
    static_file_path = '/' + static_file_path.replace(os.path.sep, '/')
    
    output_simple = None
    
    # Check if the output is a single word or a detailed result
    if '\n' in output:  # Detailed result
        # Split the output into lines and extract the first category after "Probability"
        lines = output.split('\n')
        for line in lines:
            if "Probability" in line:
                # The first category is in the line following "Probability"
                next_line_index = lines.index(line) + 1  # Get the index of the next line
                if next_line_index < len(lines):  # Check if the next line exists
                    output_simple = lines[next_line_index].strip().split()[0]  # Get the first word
                break  # No need to look further
    else:
        # Output is a single word
        output_simple = output
    
    update_stmt = update(inference_table).where(inference_table.c.id == new_id).values(output=output_simple)
    await session.execute(update_stmt)
    await session.commit()
    
    return {"status": "success", "inference": output, "link_static": static_file_path}





@router.post("/upload_image")
async def upload_image(request: Request, file: UploadFile = File(...)):
    image_directory = Path('src/static/images')
    image_directory.mkdir(parents=True, exist_ok=True)

    file_path = os.path.join(image_directory, file.filename)
    
    static_file_path = os.path.join('static', 'images', file.filename)
    static_file_path = '/' + static_file_path.replace(os.path.sep, '/')

    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await file.read()  
        await out_file.write(content)

    # Generate the URL for the saved file
    file_url = str(request.url_for('static', path=file_path))
    # file_url = f"http://127.0.0.1:8000/{str(file_path)}"
    
    #return JSONResponse(status_code=200, content={"url": file_url})
    output = await cached_inference('resnet', file_path) # temp for testing
    
    # return {"status": "success", "link": file_path, "link_static": static_file_path}
    return {"status": "success", "link": file_path, "link_static": static_file_path}


# @router.get("/database", response_model=List[InferenceCreate])
# async def get_data(session: AsyncSession = Depends(get_async_session)):
#     # Execute the query
#     async with session.begin():
#         result = await session.execute(select(inference_table))
#         db_items = result.scalars().all()

#     # Convert database objects to Pydantic model instances
#     data_items = [InferenceCreate() for item in db_items]
    
#     return db_items


@router.get("/database")
async def get_database_data(
        output: str,
        session: AsyncSession = Depends(get_async_session),
):
    try:
        # Modify the query based on whether 'output' is provided and not empty
        if output:
            query = select(inference_table).where(inference_table.c.output == output)
        else:
            query = select(inference_table)  # Select all if 'output' is not provided
        #query = select(inference_table).where(inference_table.c.output == output)
        
        result = await session.execute(query)
        data = result.all()
        data_list = []
        for row in data:
            # Convert the SQLAlchemy model instance (row) into a dictionary
            row_data = {column.key: getattr(row, column.key) for column in inference_table.c}  # Ensure this line matches your SQLAlchemy table reference
            data_list.append(row_data)

        return {
            "status": "success",
            "data": data_list,
            "details": None
        }
    except Exception as e:
        # Pass the error to developers
        raise HTTPException(status_code=500, detail={
            "status": "error",
            "data": None,
            "details": str(e)  # Providing exception details can help in debugging
        })


@router.get("/main")
async def main(session: AsyncSession = Depends(get_async_session)):
    result = await session.execute(select(1))
    return result.all()



