from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from schemas import SearchQuery, SearchResponse
from services.db_manager import db_manager
from services.video_processor import VideoProcessor
from config import SHOTS_DIR

processor: VideoProcessor | None = None

# ------------------------------------------------------------------------------
# LIFESPAN (Loading resources at start)
# ------------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor

    print("🚀 [LIFESPAN] Starting demo service...")
    
    # 1. Connect to chroma db with pre ingested trailers
    db_manager.connect()
    print("✅ [LIFESPAN] ChromaDB succesfully connected")

    # 2. Load Siglip encoder
    processor = VideoProcessor()
    print("✅ [LIFESPAN] SigLIP processor ready")

    yield

    print("🛑 [LIFESPAN] Shutting down services...")
    db_manager.disconnect()
    processor = None
    print("👋 [LIFESPAN] Services shut down succesfully")

app = FastAPI(
    title="Filmmaker Shotlist RAG API (Demo Mode)",
    version="1.0",
    description="Multimodal RAG API - Recruiter Demo",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount folder with extracted shots 
app.mount("/api/frames", StaticFiles(directory=SHOTS_DIR), name="frames")


# ------------------------------------------------------------------------------
# ENDPOINTS (Frontend + Search)
# ------------------------------------------------------------------------------

@app.get("/", response_class=FileResponse)
async def serve_index():
    return FileResponse("index.html")


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "healthy"}


@app.post("/api/search", response_model=SearchResponse)
async def search_shots(payload: SearchQuery):
    if not processor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El procesador no está inicializado.",
        )

    # 1. Convert query to embeddings for search
    query_vector = processor.encode_text_query(payload.query)

    # 2. Search on ChromaDB
    results = db_manager.search_similar(
        query_vector=query_vector,
        n_results=payload.n_results,
        movie_title=payload.movie_title,
    )

    if not results:
        return SearchResponse(query=payload.query, movie_title_filter=payload.movie_title, results=[])

    # 3. Give results proper structure
    cleaned_results = []
    for r in results:
        meta = r.get("metadata") or {}
        cleaned_meta = {
            "movie_title": meta.get("movie_title") or "Desconocido",
            "shot_index": int(meta.get("shot_index", 0) or 0),
            "timestamp": float(meta.get("timestamp", 0.0) or 0.0),
            "file_path": str(meta.get("file_path", "") or "")
        }

        cleaned_results.append({
            "id": r.get("id"),
            "score": float(r.get("score", 0.0) or 0.0),
            "metadata": cleaned_meta
        })

    return SearchResponse(
        query=payload.query,
        movie_title_filter=payload.movie_title,
        results=cleaned_results,
    )