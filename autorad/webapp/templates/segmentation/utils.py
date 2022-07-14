from __future__ import annotations

from pydantic import BaseModel


class Model(BaseModel):
    full_name: str
    modality: str
    required_sequences: tuple[str]
    tags: tuple[str]
    labels: dict[int, str]
