from fastapi import APIRouter, Request, Depends
from fastapi.templating import Jinja2Templates

from src.operations.router import get_specific_operations
import os

router = APIRouter(
    prefix="/pages",
    tags=["Pages"]
)

templates = Jinja2Templates(directory=os.path.join('src', 'templates')) # OS PATH JOIN!


@router.get("/home")
def get_base_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@router.get("/infer")
def get_base_page(request: Request):
    return templates.TemplateResponse("infer.html", {"request": request})

@router.get("/infer_google")
def get_base_page(request: Request):
    return templates.TemplateResponse("infer_gimages.html", {"request": request})

@router.get("/eval")
def get_base_page(request: Request):
    return templates.TemplateResponse("evaluate.html", {"request": request})

@router.get("/stats")
def get_base_page(request: Request):
    return templates.TemplateResponse("stats.html", {"request": request})

@router.get("/search/{operation_type}")
def get_search_page(request: Request, operations=Depends(get_specific_operations)):
    return templates.TemplateResponse("search.html", {"request": request, "operations": operations["data"]})

