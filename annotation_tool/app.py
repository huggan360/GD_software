# ---------------------
# LOCAL WEB SERVER: VIDEO VIEWER OR YOLO IMAGE ANNOTATION
# ---------------------
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from annotation_tool.settings import read_mode
from annotation_tool.store import Store, StoreError
from annotation_tool.images import add_image_routes
from annotation_tool.videos import add_video_routes


# ---------------------
# API request fields
# ---------------------
class EditorRequest(BaseModel):
    name: str = Field(min_length=1, max_length=40)


class CategoryRequest(BaseModel):
    name: str


# ---------------------
# Create the review application
# ---------------------
def create_app(settings):
    """Startar det fristående utvecklingsverktyget för klipp och klassificeringar."""
    store = Store(settings["storage"])
    static = Path(__file__).parent / "static"
    library = None

    @asynccontextmanager
    async def lifespan(app):
        """Start the video worker with the server and stop it on shutdown."""
        library.start()
        yield
        library.stop()

    app = FastAPI(title="Fridge Lab", lifespan=lifespan)
    app.state.store = store

    @app.exception_handler(StoreError)
    async def store_error(request, error):
        """Display ordinary validation errors in plain English."""
        return JSONResponse({"detail": str(error)}, status_code=error.status)

    @app.get("/")
    def home():
        """Acquisition: bildannotering för YOLO. Test och production: titta på klippen."""
        page = "images.html" if read_mode(settings) == "acquisition" else "index.html"
        return FileResponse(static / page, headers={"Cache-Control": "no-cache"})

    @app.post("/api/editors")
    def register(body: EditorRequest):
        """Register a reviewer's display name."""
        return store.register(body.name)

    @app.post("/api/categories")
    def add_category(body: CategoryRequest, x_session: str = Header()):
        """Make a new category available to every reviewer."""
        return {"name": store.add_category(body.name, x_session)}

    library = add_video_routes(app, settings, store)
    add_image_routes(app, settings, store)
    app.mount("/static", StaticFiles(directory=static), name="static")
    return app
