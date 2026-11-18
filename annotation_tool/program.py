"""Verktygets koppling till det fristående C++-programmet."""
import json
import math
import secrets
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml
from annotation_tool.zones import validate_zones

ROOT = Path(__file__).resolve().parents[1]


# Sparar en komplett JSON-fil innan den blir synlig för andra läsare.
def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


# Verktyget läser programmets inställningar. Programmet läser inga inställningar från verktyget.
def load_config(path, models_dir=None):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    config["config_path"] = str(path)
    config["zones_file"] = str((path.parent / config["zones_file"]).resolve())
    model_folder = Path(models_dir).resolve() if models_dir else (path.parent / config["models_directory"]).resolve()
    config["models_directory"] = str(model_folder)
    for section, keys in (("detector", ("weights", "hef", "hef_labels")), ("classifier", ("checkpoint",))):
        for key in keys:
            if config[section].get(key):
                config[section][key] = str(model_folder / config[section][key])
    zones_file = Path(config["zones_file"])
    if zones_file.is_file():
        for camera, zones in json.loads(zones_file.read_text()).items():
            if camera in config["cameras"]:
                config["cameras"][camera].update(zones)
    return config


# Zonfilen är vanlig kamerakonfiguration som även kan skrivas av andra program.
def save_zones(config, camera, zones):
    if camera not in config["cameras"]:
        raise ValueError("Okänd kamera.")
    zones = validate_zones(zones)
    path = Path(config["zones_file"])
    saved = json.loads(path.read_text()) if path.is_file() else {}
    saved[camera] = zones
    write_json(path, saved)
    config["cameras"][camera].update(zones)
    return zones


# Bygger rörelselistan igen efter att användaren ritat om zonerna.
def movements(tracks):
    rows = []
    for track in tracks:
        if track.get("ignored"):
            continue
        for event in track["events"]:
            rows.append({"track_id": track["id"], "direction": event["direction"], "category": track["category"], "classification_confidence": track["classification_confidence"], "start": event["event_start"], "end": event["event_end"]})
    return sorted(rows, key=lambda row: row["start"])


class ClassifierProgram:
    # Behåller både programmets inställningar och verktygets egna visningsinställningar.
    def __init__(self, config, settings):
        self.config = config
        self.settings = settings
        self.message = "C++ YOLO / ByteTrack / ViT"

    # Kör FFmpeg med en logg i verktygets egen klippmapp.
    def run_ffmpeg(self, arguments, directory):
        with (directory / "ffmpeg.log").open("a") as log:
            subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", *arguments], stdout=log, stderr=log, check=True, timeout=600)

    # Förhandsvideo, poster och produktbilder skapas bara här, aldrig av C++-programmet.
    def prepare_review(self, source, directory, result):
        source, directory = Path(source).resolve(), Path(directory).resolve()
        proxy = directory / "video.mp4"
        if source != proxy:
            width = int(self.settings.get("preview_width", 1280))
            scale = f"scale=w='min({width},trunc(iw/2)*2)':h=-2"
            self.run_ffmpeg(["-i", str(source), "-map", "0:v:0", "-an", "-vf", scale, "-c:v", "libx264", "-preset", self.settings.get("preview_preset", "veryfast"), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(proxy)], directory)
        self.run_ffmpeg(["-i", str(proxy), "-frames:v", "1", str(directory / "poster.jpg")], directory)
        width, height = result["video"]["width"], result["video"]["height"]
        for number, track in enumerate(result["tracks"], start=1):
            observations = track.get("observations", [])
            if not observations:
                continue
            best_time = track.get("best_view_time")
            if best_time is None:
                views = track.get("classification_views", [])
                best_time = views[0] if views else max(observations, key=lambda row: row["detection_confidence"])["time"]
            observation = min(observations, key=lambda row: abs(row["time"] - best_time))
            x1, y1, x2, y2 = observation["box"]
            left, top = max(0, math.floor(x1 * width)), max(0, math.floor(y1 * height))
            right, bottom = min(width, math.ceil(x2 * width)), min(height, math.ceil(y2 * height))
            if right - left < 2 or bottom - top < 2:
                continue
            # Filnamnet bygger på listans position. Även externa spår-ID:n kan då läsas utan att bli sökvägar.
            filename = f"track-{number}.jpg"
            crop = f"crop={right - left}:{bottom - top}:{left}:{top},scale=w='min(384,iw)':h=-2"
            self.run_ffmpeg(["-ss", str(best_time), "-i", str(source), "-frames:v", "1", "-vf", crop, str(directory / filename)], directory)
            track["crop_file"] = filename
        result["source"] = str(source)
        write_json(directory / "result.json", result)
        return result

    # En vanlig JSON-begäran räcker för att köra klassificeraren. Ingen annoteringskod laddas av C++.
    def run(self, source, directory, camera, zones, event_id, offset=0.0, progress=None):
        progress = progress or (lambda *_: None)
        source, directory = Path(source).resolve(), Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        validate_zones(zones)
        request, output = directory / ".request.json", directory / ".program-result.json"
        output.unlink(missing_ok=True)
        capture_id = secrets.token_hex(12)
        write_json(request, {"schema_version": 2, "capture_id": capture_id, "event_id": event_id, "captured_at": datetime.now(timezone.utc).isoformat(), "camera": camera, "video_path": str(source), "offset_seconds": offset})
        progress(10, "Analyserar klippet")
        command = [str(ROOT / "build/gd_classifier"), "--production", "--record-id", capture_id, "--request", str(request), "--output", str(output), "--config", self.config["config_path"], "--models-dir", self.config["models_directory"]]
        try:
            with (directory / "inference.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=log, check=True, timeout=600)
            result = json.loads(output.read_text())
            progress(90, "Förbereder visningen")
            self.prepare_review(source, directory, result)
        finally:
            request.unlink(missing_ok=True)
            output.unlink(missing_ok=True)
        progress(100, "Klart")
        return result
