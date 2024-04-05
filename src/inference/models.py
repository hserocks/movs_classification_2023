
from sqlalchemy import TIMESTAMP, Column, Enum, Integer, String, Table
from src.database import metadata
from enum import Enum


class ModelType(Enum):
    TYPE_1 = "resnet50"
    TYPE_2 = "vit"
    TYPE_3 = "Type 3"


inference_table = Table(
    "inference",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("model", String),  # Enum(ModelType)
    Column("link", String),
    Column("date", TIMESTAMP, nullable=True),
    # Column("type", String, nullable=True),
    Column("output", String, nullable=True),
)
