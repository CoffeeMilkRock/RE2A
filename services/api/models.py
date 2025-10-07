from sqlalchemy import Column, Integer, String, Text, Float
from sqlalchemy.orm import Mapped, mapped_column
from .db import Base


class Character(Base):
    __tablename__ = "characters"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    style: Mapped[str] = mapped_column(Text)
    objective: Mapped[str] = mapped_column(Text)


class Property(Base):
    __tablename__ = "properties"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    location: Mapped[str] = mapped_column(String(200))
    property_type: Mapped[str] = mapped_column(String(50))
    bedrooms: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    description: Mapped[str] = mapped_column(Text)
    raw_json: Mapped[str] = mapped_column(Text) # original schema blob