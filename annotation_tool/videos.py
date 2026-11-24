# ---------------------
# Utvecklingsverktyg (test/production): titta på klipp och modellernas resultat, kör om på nya modeller.
# ---------------------
import json
import re
import secrets
import shutil
import threading
import time
from pathlib import Path

from fastapi import File, Form, Header, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from annotation_tool.store import StoreError
from annotation_tool.program import ClassifierProgram, load_config, write_json
from annotation_tool.progress import read_progress
from annotation_tool.settings import read_mode

VIDEO_ID = re.compile(r"[0-9a-f]{24}")
EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".h264", ".m4v"}


# ---------------------
# API request fields
# ---------------------
class ReclassifyRequest(BaseModel):
    all_videos: bool = False


# ---------------------
# Video storage and processing queue
# ---------------------
class VideoLibrary:
    """One folder per clip; meta.json holds queue state, result.json the model output."""

    def __init__(self, settings):
        self.settings = settings
        self.config_path = settings["video_config"]
        self.config = load_config(self.config_path)
        self.root = Path(settings["storage"]) / "videos"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.program = None
        self.message = "Video models load when the first clip is processed."
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.work, daemon=True, name="video-pipeline")

    def folder(self, video_id):
        if not VIDEO_ID.fullmatch(video_id) or not (self.root / video_id / "meta.json").is_file():
            raise StoreError("Video not found.", 404)
        return self.root / video_id

    def meta(self, video_id):
        return json.loads((self.folder(video_id) / "meta.json").read_text())

    def update(self, video_id, **values):
        with self.lock:
            path = self.folder(video_id) / "meta.json"
            meta = json.loads(path.read_text())
            meta.update(values)
            write_json(path, meta)
            return meta

    # ---------------------
    # Back up and reclassify videos
    # ---------------------
    def reclassify(self, editor, video_ids=None):
        """Back up predictions, then queue clips for fresh model outputs."""
        batch = secrets.token_hex(12)
        backup = self.root.parent / "reclassification_backups" / batch
        queued = []
        skipped = []
        with self.lock:
            ids = video_ids if video_ids is not None else [m["id"] for m in self.items()]
            for video_id in ids:
                folder = self.folder(video_id)
                meta = self.meta(video_id)
                if meta["status"] in ("queued", "running"):
                    skipped.append(video_id)
                    continue
                destination = backup / video_id
                destination.mkdir(parents=True)
                paths = [folder / name for name in ("meta.json", "result.json")]
                paths.extend(folder.glob("track-*.jpg"))
                for path in paths:
                    if path.is_file():
                        shutil.copy2(path, destination / path.name)
                meta.update(
                    status="queued",
                    progress=0,
                    message="Waiting for reclassification",
                    reclassification_batch=batch,
                    reclassified_by=editor,
                )
                write_json(folder / "meta.json", meta)
                queued.append(video_id)
            backup.mkdir(parents=True, exist_ok=True)
            report = {
                "batch": batch,
                "queued": len(queued),
                "video_ids": queued,
                "skipped_active": len(skipped),
                "skipped_ids": skipped,
                "backup_directory": str(backup),
                "editor": editor,
                "created_at": time.time(),
                "models": {
                    "backend": "hailo",
                    "yolo": self.config["detector"]["hef"],
                    "vit": self.config["classifier"].get("checkpoint"),
                },
            }
            write_json(backup / "batch.json", report)
        return report

    def add(self, source, filename, camera, event_id, editor, move=False):
        """Copy (or move) a clip in and queue it for the pipeline."""
        if camera not in self.config["cameras"]:
            raise StoreError(f"Camera must be one of: {', '.join(self.config['cameras'])}.")
        suffix = Path(filename).suffix.lower()
        if suffix not in EXTENSIONS:
            raise StoreError("Upload an MP4, MOV, MKV, AVI, WEBM or H.264 video.")
        video_id = secrets.token_hex(12)
        folder = self.root / video_id
        folder.mkdir()
        target = folder / f"original{suffix}"
        (shutil.move if move else shutil.copyfile)(source, target)
        write_json(
            folder / "meta.json",
            {
                "id": video_id,
                "filename": Path(filename).name[:200],
                "camera": camera,
                "event_id": event_id or f"upload-{video_id[:8]}",
                "created": time.time(),
                "editor": editor,
                "status": "queued",
                "progress": 0,
                "message": "Waiting",
                "original": target.name,
            },
        )
        return video_id

    def items(self):
        rows = []
        for path in self.root.glob("*/meta.json"):
            if not VIDEO_ID.fullmatch(path.parent.name):
                continue  # Half-published or temporary folders.
            meta = json.loads(path.read_text())
            meta["processing"] = read_progress(meta)
            result = path.parent / "result.json"
            if result.is_file():
                document = json.loads(result.read_text())
                meta["movements"] = len(document.get("movements", []))
                meta["tracks"] = len(document.get("tracks", []))
            rows.append(meta)
        return sorted(rows, key=lambda row: row["created"], reverse=True)

    # ---------------------
    # ONE BACKGROUND WORKER: THE PI HAS ONE HAILO-8
    # ---------------------
    def start(self):
        for meta in self.items():
            if meta["status"] == "running":
                self.update(meta["id"], status="queued", progress=0, message="Restarted")
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)

    # ---------------------
    # Background processing loop
    # ---------------------
    def work(self):
        while not self.stop_event.is_set():
            queued = [row for row in self.items() if row["status"] == "queued"]
            if not queued:
                self.stop_event.wait(0.5)
                continue
            meta = queued[-1]  # Oldest first.
            folder = self.root / meta["id"]
            try:
                self.update(meta["id"], status="running", message="Loading models")
                if self.program is None:
                    self.program = ClassifierProgram(self.config, self.settings)
                    self.message = self.program.message
                started = time.monotonic()
                source = folder / meta["original"]
                self.program.run(source, folder, meta["camera"], meta["event_id"], progress=lambda value, text: self.update(meta["id"], progress=value, message=text))
                self.update(
                    meta["id"],
                    status="done",
                    progress=100,
                    message=f"Processed in {time.monotonic() - started:.1f} s",
                )
            except Exception as error:
                self.update(meta["id"], status="failed", message=str(error)[:1000])


# ---------------------
# Video API routes
# ---------------------
def add_video_routes(app, settings, store):
    """Register the video API."""
    library = VideoLibrary(settings)
    app.state.videos = library
    limit = library.config["video"]["max_upload_mb"] * 1024 * 1024

    def editor(token):
        with store.connect(write=True) as connection:
            return store.editor(token, connection)

    @app.get("/api/videos")
    def list_videos():
        config = library.config
        return {
            "items": library.items(),
            "cameras": config["cameras"],
            "mode": read_mode(settings),
            "model_message": library.message,
        }

    @app.post("/api/videos")
    def upload_video(
        file: UploadFile = File(),
        camera: str = Form(),
        event_id: str = Form(default=""),
        x_session: str = Header(),
    ):
        name = editor(x_session)
        temporary = library.root / f".upload-{secrets.token_hex(6)}"
        try:
            size = 0
            with temporary.open("wb") as handle:
                while chunk := file.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > limit:
                        raise StoreError("Video exceeds the configured upload limit.")
                    handle.write(chunk)
            return {
                "id": library.add(
                    temporary,
                    file.filename or "upload.mp4",
                    camera,
                    event_id.strip(),
                    name,
                    move=True,
                )
            }
        finally:
            temporary.unlink(missing_ok=True)
            file.file.close()

    @app.post("/api/videos/reclassify")
    def reclassify_all(body: ReclassifyRequest, x_session: str = Header()):
        name = editor(x_session)
        if not body.all_videos:
            raise StoreError("Set all_videos=true to explicitly reclassify the whole library.")
        return library.reclassify(name)

    @app.get("/api/videos/{video_id}")
    def video_detail(video_id: str):
        folder = library.folder(video_id)
        value = {"meta": library.meta(video_id)}
        value["meta"]["processing"] = read_progress(value["meta"])
        path = folder / "result.json"
        value["result"] = json.loads(path.read_text()) if path.is_file() else None
        return value

    @app.get("/api/videos/{video_id}/files/{name}")
    def video_file(video_id: str, name: str):
        if not re.fullmatch(r"video\.mp4|poster\.jpg|track-\d+\.jpg", name):
            raise StoreError("File not found.", 404)
        path = library.folder(video_id) / name
        if not path.is_file():
            raise StoreError("File not found.", 404)
        return FileResponse(path)

    @app.post("/api/videos/{video_id}/rerun")
    def rerun(video_id: str, x_session: str = Header()):
        name = editor(x_session)
        if library.meta(video_id)["status"] in ("queued", "running"):
            raise StoreError("This clip is already queued.", 409)
        result = library.reclassify(name, [video_id])
        if not result["queued"]:
            raise StoreError("This clip is already queued.", 409)
        return library.meta(video_id)

    return library
