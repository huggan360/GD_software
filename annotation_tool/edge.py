"""Läser färdiga körningar från edge utan att ändra dess databas eller originalfiler."""
import json
import secrets
import shutil
import sqlite3
import time
from pathlib import Path

from annotation_tool.program import write_json
from annotation_tool.store import StoreError


# Kopierar nya klipp till verktygets egen kö. Klassificeringarna följer med från edge.
def import_edge_videos(database, library, editor):
    if not database:
        raise StoreError("Ange edge_database i annoteringsverktygets konfiguration.", 409)
    path = Path(database).resolve()
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute("SELECT predictions.run_id, predictions.result_json, captures.input_json FROM predictions JOIN captures ON captures.capture_id = predictions.capture_id JOIN inference_runs ON inference_runs.run_id = predictions.run_id WHERE inference_runs.status = 'completed' ORDER BY predictions.created_at").fetchall()
    finally:
        connection.close()

    imported, missing = 0, 0
    with library.lock:
        known = {item.get("edge_run_id") for item in library.items()}
        for row in rows:
            if row["run_id"] in known:
                continue
            request = json.loads(row["input_json"])
            result = json.loads(row["result_json"])
            source = Path(request["video_path"])
            if not source.is_file() or request["camera"] not in library.config["cameras"]:
                missing += 1
                continue
            video_id = secrets.token_hex(12)
            staging = library.root / (".edge-" + video_id)
            staging.mkdir()
            try:
                original = "original" + source.suffix.lower()
                shutil.copyfile(source, staging / original)
                write_json(staging / ".edge-result.json", result)
                write_json(staging / "meta.json", {"id": video_id, "filename": f"{request['event_id']}-{request['camera']}{source.suffix}", "camera": request["camera"], "event_id": request["event_id"], "created": time.time(), "editor": editor, "status": "queued", "progress": 0, "message": "Förbereder klipp från edge", "original": original, "edge_run_id": row["run_id"]})
                staging.rename(library.root / video_id)
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                raise
            known.add(row["run_id"])
            imported += 1
    return {"imported": imported, "missing": missing}
