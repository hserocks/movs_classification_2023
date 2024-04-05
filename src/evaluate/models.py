
from sqlalchemy import TIMESTAMP, Column, Enum, Integer, String, Table, Boolean
from src.database import metadata
from enum import Enum


class ModelType(Enum):
    TYPE_1 = "resnet50"
    TYPE_2 = "vit"
    TYPE_3 = "Type 3"


evaluate_table = Table(
    "evaluate",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("model", String),  # Enum(ModelType)
    Column("data", String),
    Column("evaluate_only", Boolean),
    Column("date", TIMESTAMP, nullable=True),
    Column("type", String, nullable=True),
)

evaluate_fmodels = Table(
    "evaluate_fmodels",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("features", String),  # Enum(ModelType)
    Column("fmodel", String),
    Column("evaluate_only", Boolean),
    Column("date", TIMESTAMP, nullable=True),
    Column("type", String, nullable=True),
)
