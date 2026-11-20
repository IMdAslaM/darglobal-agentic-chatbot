"""
app.py - FastAPI Application for DarGlobal & Wasalt Real Estate AI Chatbot.

Provides:
  - RESTful Chat API with RAG retrieval via OpenRouter
  - Property search & filter API
  - On-demand background scraper trigger
  - Static file serving for the interactive Web Chat UI
"""

import os
import json
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, status
from typing import Literal
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from rag import PropertyRAG, DEFAULT_MODELS
from scraper import run_scraper

load_dotenv()

logger = logging.getLogger("app")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated on_event startup/shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("DarGlobal & Wasalt Real Estate AI starting up…")
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DarGlobal & Wasalt Real Estate AI",
    description=(
        "Agentic RAG Chatbot for Luxury Branded Residences (DarGlobal) "
        "& Saudi Real Estate (Wasalt). Powered by OpenRouter free models."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

# CORS – allow all origins in development; tighten for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# RAG Engine (singleton)
# ---------------------------------------------------------------------------
rag_engine = PropertyRAG()

# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User query or message")
    history: Optional[List[ChatMessage]] = Field(default=[], description="Recent conversation turns")
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional property filters: {portal, city, property_type}",
    )


class ChatResponse(BaseModel):
    response: str
    properties: List[Dict[str, Any]]
    model: str
    status: str
    error_detail: Optional[str] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def serve_index():
    """Serves the main single-page web chat application."""
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file, media_type="text/html")
    return JSONResponse(
        status_code=200,
        content={"message": "DarGlobal & Wasalt AI Chatbot is running. Add static/index.html to serve the UI."},
    )


@app.get("/api/health", tags=["System"])
async def health_check():
    """System health & configuration status."""
    return {
        "status": "healthy",
        "total_properties": len(rag_engine.properties),
        "openrouter_configured": bool(rag_engine.client),
        "primary_model": rag_engine.primary_model,
        "version": "1.1.0",
    }


@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(request: ChatRequest):
    """
    RAG chat endpoint:
    - Retrieves context from DarGlobal and Wasalt listings.
    - Generates AI response using OpenRouter free models.
    """
    history_payload = (
        [{"role": h.role, "content": h.content} for h in request.history]
        if request.history
        else []
    )
    result = rag_engine.chat(
        user_message=request.message,
        history=history_payload,
        filters=request.filters,
    )
    return result


@app.post("/api/chat/stream", tags=["Chat"])
async def chat_stream_endpoint(request: ChatRequest):
    """
    Streaming SSE chat endpoint — delivers tokens one-by-one like ChatGPT.
    Frontend reads via fetch + ReadableStream.
    """
    history_payload = (
        [{"role": h.role, "content": h.content} for h in request.history]
        if request.history
        else []
    )

    async def event_generator():
        # Build RAG context (fast, synchronous)
        matched, messages, corrected = rag_engine.build_chat_context(
            user_message=request.message,
            history=history_payload,
            filters=request.filters,
        )

        # Send matched properties first so sidebar updates immediately
        yield f"data: {json.dumps({'type': 'properties', 'data': matched})}\n\n"

        # Offline / no client fallback
        if not rag_engine.client:
            fallback = rag_engine._generate_conversational_fallback(
                request.message, matched, corrected
            )
            yield f"data: {json.dumps({'type': 'token', 'data': fallback})}\n\n"
            yield f"data: {json.dumps({'type': 'done',  'model': 'local-fallback'})}\n\n"
            return

        # Try each model with streaming
        models_to_try = [rag_engine.primary_model] + [
            m for m in DEFAULT_MODELS if m != rag_engine.primary_model
        ]

        for model in models_to_try:
            try:
                loop = asyncio.get_event_loop()
                queue: asyncio.Queue = asyncio.Queue()

                def _stream_in_thread(mdl=model):
                    """Runs synchronous OpenAI stream in a thread, pushes to async queue."""
                    try:
                        stream = rag_engine.client.chat.completions.create(
                            model=mdl,
                            messages=messages,  # type: ignore[arg-type]
                            temperature=0.75,
                            max_tokens=1400,
                            stream=True,
                        )
                        for chunk in stream:
                            delta = chunk.choices[0].delta.content
                            if delta:
                                loop.call_soon_threadsafe(
                                    queue.put_nowait, ("token", delta)
                                )
                        loop.call_soon_threadsafe(queue.put_nowait, ("done", mdl))
                    except Exception as exc:
                        loop.call_soon_threadsafe(
                            queue.put_nowait, ("error", str(exc))
                        )

                # Launch thread
                loop.run_in_executor(None, _stream_in_thread)

                got_tokens = False
                while True:
                    kind, payload = await queue.get()
                    if kind == "token":
                        got_tokens = True
                        yield f"data: {json.dumps({'type': 'token', 'data': payload})}\n\n"
                    elif kind == "done":
                        yield f"data: {json.dumps({'type': 'done', 'model': payload})}\n\n"
                        return
                    elif kind == "error":
                        logger.warning(f"Stream model {model} error: {payload}")
                        if got_tokens:
                            # Partial response already sent — close cleanly
                            yield f"data: {json.dumps({'type': 'done', 'model': model})}\n\n"
                            return
                        break  # Try next model

            except Exception as exc:
                logger.warning(f"Stream setup error for {model}: {exc}")
                continue

        # All models failed — use local fallback
        fallback = rag_engine._generate_conversational_fallback(
            request.message, matched, corrected
        )
        yield f"data: {json.dumps({'type': 'token', 'data': fallback})}\n\n"
        yield f"data: {json.dumps({'type': 'done',  'model': 'local-fallback'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":  "keep-alive",
        },
    )


@app.get("/api/properties", tags=["Properties"])
async def list_properties(
    portal: Optional[str] = Query(None, description="DarGlobal or Wasalt"),
    city: Optional[str] = Query(None, description="e.g. Dubai, Riyadh, Jeddah, London"),
    property_type: Optional[str] = Query(None, description="e.g. Villa, Apartment, Penthouse"),
    search: Optional[str] = Query(None, description="Keyword search"),
    limit: int = Query(50, ge=1, le=200, description="Max results to return"),
):
    """List and filter the property catalogue."""
    items = rag_engine.search_properties(search, top_k=limit) if search else rag_engine.properties[:limit]

    filtered: List[Dict[str, Any]] = []
    for p in items:
        if portal and p.get("developer_or_portal", "").lower() != portal.lower():
            continue
        if city and city.lower() not in p.get("city", "").lower():
            continue
        if property_type and property_type.lower() not in p.get("property_type", "").lower():
            continue
        filtered.append(p)

    return {"total": len(filtered), "properties": filtered}


@app.get("/api/properties/{prop_id}", tags=["Properties"])
async def get_property_by_id(prop_id: str):
    """Fetch a single property by its ID."""
    for p in rag_engine.properties:
        if p.get("id") == prop_id:
            return p
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Property '{prop_id}' not found.")


@app.post("/api/scrape", tags=["System"])
async def trigger_scrape(
    background_tasks: BackgroundTasks,
    source: Literal["all", "darglobal", "wasalt"] = "all",
):
    """Trigger a background scraper run to refresh the property dataset."""

    def _bg_scrape():
        try:
            run_scraper(source=source)
            count = rag_engine.reload_data()
            logger.info(f"Background scrape complete. Dataset now has {count} properties.")
        except Exception as exc:
            logger.error(f"Background scrape failed: {exc}")

    background_tasks.add_task(_bg_scrape)
    return {"message": f"Property scrape initiated for source='{source}'.", "source": source}


# ---------------------------------------------------------------------------
# Dev entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 7860))
    host = os.getenv("HOST", "0.0.0.0")
    debug = os.getenv("DEBUG", "false").lower() == "true"
    uvicorn.run("app:app", host=host, port=port, reload=debug)
