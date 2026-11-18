"""Let the unchanged review tool use the C++ classifier."""
import json
import secrets
import subprocess
from pathlib import Path

import yaml
from production.zones import validate_zones

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_config(path, models_dir=None):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    config["config_path"] = str(path)
    for key in ("acquisition_directory", "production_directory", "zones_file"):
        config[key] = str((path.parent / config[key]).resolve())
    root = Path(models_dir).resolve() if models_dir else (path.parent / config["models_directory"]).resolve()
    config["models_directory"] = str(root)
    for section, keys in (("detector", ("weights", "hef", "hef_labels")), ("classifier", ("checkpoint",))):
        for key in keys:
            if config[section].get(key):
                config[section][key] = str(root / config[section][key])
    zones = Path(config["zones_file"])
    if zones.is_file():
        for camera, values in json.loads(zones.read_text()).items():
            if camera in config["cameras"]:
                config["cameras"][camera].update(values)
    for values in config["cameras"].values():
        validate_zones(values)
    return config


def save_zones(config, camera, zones):
    if camera not in config["cameras"]:
        raise ValueError("Unknown camera.")
    zones = validate_zones(zones)
    path = Path(config["zones_file"])
    saved = json.loads(path.read_text()) if path.is_file() else {}
    saved[camera] = zones
    write_json(path, saved)
    config["cameras"][camera].update(zones)
    return zones


def movements(tracks):
    rows = [{"track_id": track["id"], "direction": event["direction"], "category": track["category"], "classification_confidence": track["classification_confidence"], "start": event["event_start"], "end": event["event_end"]} for track in tracks if not track.get("ignored") for event in track["events"]]
    return sorted(rows, key=lambda row: row["start"])


class VideoPipeline:
    def __init__(self, config):
        self.config = config

    @property
    def message(self):
        return "C++ YOLO / ByteTrack / ViT"

    def release(self):
        # Every subprocess releases its models and accelerator when it exits.
        pass

    def run(self, source, directory, camera, zones, event_id, offset=0.0, progress=None):
        progress = progress or (lambda *_: None)
        source, directory = Path(source).resolve(), Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        validate_zones(zones)
        binary = ROOT / "build/gd_classifier"
        if not binary.is_file():
            raise RuntimeError("Build the classifier first: bash compile/build.sh")
        progress(5, "Preparing review playback")
        # Browser conversion is only for uploaded review clips, never the production handoff.
        proxy = directory / "video.mp4"
        if source != proxy:
            width = int(self.config["video"]["proxy_width"])
            command = ["ffmpeg", "-nostdin", "-y", "-i", str(source), "-map", "0:v:0", "-an", "-vf", f"scale=w='min({width},trunc(iw/2)*2)':h=-2", "-c:v", "libx264", "-preset", self.config["video"]["proxy_preset"], "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(proxy)]
            with (directory / "ffmpeg.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=log, check=True, timeout=600)
        capture_id = secrets.token_hex(12)
        request = directory / ".review-request.json"
        output = directory / ".review-result.json"
        output.unlink(missing_ok=True)
        write_json(request, {"schema_version": 2, "capture_id": capture_id, "event_id": event_id, "captured_at": "review", "camera": camera, "video_path": str(proxy), "offset_seconds": offset})
        progress(15, "Tracking and classifying products")
        try:
            with (directory / "inference.log").open("w") as log:
                subprocess.run([str(binary), "--production", "--managed-retention", "--record-id", capture_id, "--request", str(request), "--output", str(output), "--config", self.config["config_path"], "--models-dir", self.config["models_directory"]], stdout=log, stderr=log, check=True, timeout=600)
            result = json.loads(output.read_text())
            result["source"] = str(source)
            write_json(directory / "result.json", result)
        finally:
            request.unlink(missing_ok=True)
            output.unlink(missing_ok=True)
        progress(100, "Ready")
        return result
