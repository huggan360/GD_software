# ---------------------
# Bildinsamling för YOLO: ta eller ladda upp foton, rita rutor och bygg datasetet.
# Inga modeller körs på bilderna. Kameran styrs av edge, aldrig av C++-programmet.
# ---------------------
import csv
import json
import re
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
import zipfile
from pathlib import Path

import yaml
from fastapi import File, Header, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from annotation_tool.program import write_json
from annotation_tool.settings import read_mode
from annotation_tool.store import StoreError, category_name

IMAGE_ID = re.compile(r"[0-9a-f]{24}")
EXTENSIONS = {".jpg": ".jpg", ".jpeg": ".jpg", ".png": ".png"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
# Samma kolumner som GD_tracking/utilities/synthetic_data.py skriver.
CSV_FIELDS = ["image", "product_name", "class_id", "source_image", "background_frame", "variant",
              "xmin", "ymin", "xmax", "ymax", "width", "height", "split"]


# ---------------------
# API request fields
# ---------------------
class BoxRequest(BaseModel):
    category: str
    box: list[float] = Field(min_length=4, max_length=4)


class AnnotationRequest(BaseModel):
    boxes: list[BoxRequest] = Field(max_length=200)


# ---------------------
# One folder per photo: the image, a thumbnail, meta.json and annotation.json once saved
# ---------------------
class ImageLibrary:
    def __init__(self, settings):
        self.settings = settings
        self.root = Path(settings["storage"]) / "images"
        self.dataset = Path(settings["storage"]) / "yolo_dataset"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.camera_lock = threading.Lock()

    def folder(self, image_id):
        if not IMAGE_ID.fullmatch(image_id) or not (self.root / image_id / "meta.json").is_file():
            raise StoreError("Image not found.", 404)
        return self.root / image_id

    def meta(self, image_id):
        return json.loads((self.folder(image_id) / "meta.json").read_text())

    def annotation(self, image_id):
        path = self.folder(image_id) / "annotation.json"
        return json.loads(path.read_text()) if path.is_file() else None

    def items(self):
        rows = []
        for path in self.root.glob("*/meta.json"):
            if not IMAGE_ID.fullmatch(path.parent.name):
                continue
            meta = json.loads(path.read_text())
            annotation = path.parent / "annotation.json"
            if annotation.is_file():
                document = json.loads(annotation.read_text())
                meta.update(annotated=True, boxes=len(document["boxes"]), annotated_by=document["editor"])
            else:
                meta.update(annotated=False, boxes=0)
            rows.append(meta)
        return sorted(rows, key=lambda row: row["created"], reverse=True)

    # ---------------------
    # Add a photo (camera or upload)
    # ---------------------
    def add(self, source, filename, editor, origin):
        suffix = EXTENSIONS.get(Path(filename).suffix.lower())
        if suffix is None:
            raise StoreError("Upload JPG or PNG images.")
        image_id = secrets.token_hex(12)
        staging = self.root / f".add-{image_id}"
        staging.mkdir()
        try:
            target = staging / f"image{suffix}"
            with Image.open(source) as picture:
                if picture.getexif().get(0x0112, 1) != 1:
                    # Telefonbilder: spara pixlarna rättvända så att rutorna stämmer i YOLO.
                    upright = ImageOps.exif_transpose(picture)
                    upright.save(target, quality=95)
                else:
                    upright = picture
                    shutil.copyfile(source, target)
                width, height = upright.size
                thumbnail = upright.convert("RGB")
                thumbnail.thumbnail((320, 320))
                thumbnail.save(staging / "thumb.jpg", quality=80)
            write_json(staging / "meta.json", {
                "id": image_id, "filename": Path(filename).name[:200], "image": target.name,
                "width": width, "height": height, "created": time.time(),
                "editor": editor, "source": origin,
            })
            staging.rename(self.root / image_id)
        except StoreError:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        except Exception as error:
            shutil.rmtree(staging, ignore_errors=True)
            raise StoreError(f"Could not read {Path(filename).name}: {error}") from error
        return image_id

    # ---------------------
    # Camera: edge takes the photo, only in acquisition mode
    # ---------------------
    def camera_state(self):
        command = self.settings.get("capture_command")
        if not command:
            return {"available": False, "reason": "No capture_command in config.yaml; upload images instead."}
        mode = read_mode(self.settings)
        if mode != "acquisition":
            return {"available": False, "reason": f"The fridge is in {mode} mode. Run: smartfridge mode acquisition"}
        return {"available": True, "reason": ""}

    def capture(self, editor):
        state = self.camera_state()
        if not state["available"]:
            raise StoreError(state["reason"], 409)
        if not self.camera_lock.acquire(blocking=False):
            raise StoreError("The camera is already taking a photo.", 409)
        staging = self.root / f".capture-{secrets.token_hex(6)}"
        try:
            staging.mkdir()
            photo = staging / "photo.jpg"
            finished = subprocess.run([*map(str, self.settings["capture_command"]), str(photo)],
                                      capture_output=True, text=True, timeout=60)
            if finished.returncode != 0 or not photo.is_file():
                raise StoreError(f"Camera failed: {(finished.stderr or finished.stdout).strip()[-400:]}", 500)
            return self.add(photo, time.strftime("photo-%Y%m%d-%H%M%S.jpg"), editor, "camera")
        except subprocess.TimeoutExpired as error:
            raise StoreError("The camera did not answer within 60 s.", 500) from error
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            self.camera_lock.release()

    # ---------------------
    # Save boxes, then rebuild the YOLO dataset
    # ---------------------
    def save(self, image_id, boxes, categories, editor):
        meta = self.meta(image_id)
        width, height = meta["width"], meta["height"]
        rows = []
        for box in boxes:
            name = category_name(box.category)
            if name not in categories:
                raise StoreError(f"Unknown item {name}. Add it to the list first.")
            x1, y1, x2, y2 = box.box
            x1, x2 = sorted((max(0, min(width, round(x1))), max(0, min(width, round(x2)))))
            y1, y2 = sorted((max(0, min(height, round(y1))), max(0, min(height, round(y2)))))
            if x2 - x1 < 2 or y2 - y1 < 2:
                raise StoreError("A box is too small. Delete it or draw it again.")
            rows.append({"category": name, "box": [x1, y1, x2, y2]})
        document = {"boxes": rows, "editor": editor, "saved_at": time.time()}
        with self.lock:
            write_json(self.folder(image_id) / "annotation.json", document)
            self.build_dataset()
        return document

    def delete(self, image_id):
        with self.lock:
            folder = self.folder(image_id)
            shutil.rmtree(folder)
            self.build_dataset()

    def build_dataset(self):
        """Skriv om hela datasetet: images/, labels/, annotations.csv, dataset.yaml, classes.txt."""
        annotated = []
        for meta in self.items():
            if meta["annotated"]:
                annotated.append((meta, json.loads((self.root / meta["id"] / "annotation.json").read_text())))
        annotated.reverse()  # Äldst först.
        names = sorted({box["category"] for _, document in annotated for box in document["boxes"]})
        class_ids = {name: number for number, name in enumerate(names)}
        validation = float(self.settings.get("validation_fraction", 0.2))

        output = self.dataset.with_name(".yolo_dataset-new")
        shutil.rmtree(output, ignore_errors=True)
        for split in ("train", "val"):
            (output / "images" / split).mkdir(parents=True)
            (output / "labels" / split).mkdir(parents=True)
        with (output / "annotations.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for meta, document in annotated:
                # Bilden får alltid samma split, oavsett hur många som annoterats.
                split = "val" if int(meta["id"][:8], 16) % 1000 < validation * 1000 else "train"
                source = self.root / meta["id"] / meta["image"]
                image_path = Path("images") / split / (meta["id"] + source.suffix)
                try:
                    (output / image_path).hardlink_to(source)
                except OSError:
                    shutil.copy2(source, output / image_path)
                width, height = meta["width"], meta["height"]
                lines = []
                common = {"image": str(image_path), "source_image": meta["filename"], "background_frame": "",
                          "variant": "real", "width": width, "height": height, "split": split}
                for box in document["boxes"]:
                    left, top, right, bottom = box["box"]
                    class_id = class_ids[box["category"]]
                    lines.append(f"{class_id} {(left + right) / (2 * width):.6f} {(top + bottom) / (2 * height):.6f} "
                                 f"{(right - left) / width:.6f} {(bottom - top) / height:.6f}\n")
                    writer.writerow(dict(common, product_name=box["category"], class_id=class_id,
                                         xmin=left, ymin=top, xmax=right, ymax=bottom))
                if not document["boxes"]:
                    # Tom bild: negativt exempel med tom labelfil, som i GD_tracking.
                    writer.writerow(dict(common, product_name="", class_id="", xmin="", ymin="", xmax="", ymax=""))
                (output / "labels" / split / (meta["id"] + ".txt")).write_text("".join(lines))
        settings = {"path": str(self.dataset.resolve()), "train": "images/train", "val": "images/val", "names": names}
        (output / "dataset.yaml").write_text(yaml.safe_dump(settings, allow_unicode=True, sort_keys=False), encoding="utf-8")
        (output / "classes.txt").write_text("".join(name + "\n" for name in names), encoding="utf-8")
        old = self.dataset.with_name(".yolo_dataset-old")
        shutil.rmtree(old, ignore_errors=True)
        if self.dataset.exists():
            self.dataset.rename(old)
        output.rename(self.dataset)
        shutil.rmtree(old, ignore_errors=True)

    def export(self):
        """Zip the dataset; dataset.yaml without path, so YOLO uses the folder it is unzipped to."""
        handle = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        handle.close()
        with self.lock:
            if not (self.dataset / "annotations.csv").is_file():
                self.build_dataset()
            with zipfile.ZipFile(handle.name, "w") as archive:
                for path in sorted(self.dataset.rglob("*")):
                    if path.is_file() and path.name != "dataset.yaml":
                        archive.write(path, "yolo_dataset/" + str(path.relative_to(self.dataset)))
                settings = yaml.safe_load((self.dataset / "dataset.yaml").read_text())
                settings.pop("path")
                archive.writestr("yolo_dataset/dataset.yaml", yaml.safe_dump(settings, allow_unicode=True, sort_keys=False))
        return Path(handle.name)


# ---------------------
# Image API routes
# ---------------------
def add_image_routes(app, settings, store):
    library = ImageLibrary(settings)
    app.state.images = library

    def editor(token):
        with store.connect(write=True) as connection:
            return store.editor(token, connection)

    @app.get("/api/images")
    def list_images():
        items = library.items()
        return {
            "items": items,
            "categories": store.categories(),
            "camera": library.camera_state(),
            "annotated": sum(item["annotated"] for item in items),
            "mode": read_mode(settings),
        }

    @app.post("/api/images")
    def upload_images(files: list[UploadFile] = File(), x_session: str = Header()):
        name = editor(x_session)
        added = []
        for file in files:
            temporary = library.root / f".upload-{secrets.token_hex(6)}"
            try:
                size = 0
                with temporary.open("wb") as handle:
                    while chunk := file.file.read(1024 * 1024):
                        size += len(chunk)
                        if size > MAX_UPLOAD_BYTES:
                            raise StoreError(f"{file.filename} is larger than 50 MB.")
                        handle.write(chunk)
                added.append(library.add(temporary, file.filename or "upload.jpg", name, "upload"))
            finally:
                temporary.unlink(missing_ok=True)
                file.file.close()
        return {"ids": added}

    @app.post("/api/images/capture")
    def capture(x_session: str = Header()):
        return {"id": library.capture(editor(x_session))}

    @app.get("/api/images/dataset")
    def download_dataset(x_session: str = Header()):
        from starlette.background import BackgroundTask

        editor(x_session)
        path = library.export()
        return FileResponse(path, filename="yolo_dataset.zip", media_type="application/zip",
                            background=BackgroundTask(path.unlink, missing_ok=True))

    @app.get("/api/images/{image_id}")
    def image_detail(image_id: str):
        return {"meta": library.meta(image_id), "annotation": library.annotation(image_id)}

    @app.get("/api/images/{image_id}/file")
    def image_file(image_id: str):
        meta = library.meta(image_id)
        return FileResponse(library.folder(image_id) / meta["image"])

    @app.get("/api/images/{image_id}/thumb")
    def image_thumb(image_id: str):
        return FileResponse(library.folder(image_id) / "thumb.jpg")

    @app.put("/api/images/{image_id}/annotation")
    def save_annotation(image_id: str, body: AnnotationRequest, x_session: str = Header()):
        name = editor(x_session)
        return library.save(image_id, body.boxes, set(store.categories()), name)

    @app.delete("/api/images/{image_id}")
    def delete_image(image_id: str, x_session: str = Header()):
        editor(x_session)
        library.delete(image_id)
        return {"deleted": image_id}

    return library
