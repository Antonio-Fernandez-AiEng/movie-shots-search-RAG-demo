# schemas.py
from pydantic import BaseModel, Field
from typing import List

# --- PETICIONES (Requests) ---


class SearchQuery(BaseModel):
    query: str = Field(..., description="Description of the search query (e.g., 'dramatic close-up')")
    n_results: int = Field(default=5, ge=1, le=50, description="Result number to return (default 5, max 50)")
    movie_title: str | None = Field(default=None, description="Movie title filter (optional)")


# --- RESPONSES ---

class ShotMetadata(BaseModel):
    movie_title: str | None = "Desconocido"
    shot_index: int | None = Field(default=0)
    timestamp: float | None = Field(default=0.0)
    file_path: str | None = Field(default="")
    

class ShotResult(BaseModel):
    id: str
    score: float
    metadata: ShotMetadata | None = Field(default_factory=ShotMetadata)

class SearchResponse(BaseModel):
    query: str
    movie_title_filter: str | None = None
    results: List[ShotResult]