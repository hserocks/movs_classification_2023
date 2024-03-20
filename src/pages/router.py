from fastapi import APIRouter, Request, Depends
from fastapi.templating import Jinja2Templates

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
