import re
from pathlib import Path


# Läser vad programmet faktiskt arbetar med. Gamla resultat används inte som framsteg.
def read_progress(folder, meta):
    if meta["status"] != "running":
        return None
    folder = Path(folder)
    if meta.get("progress", 0) >= 90 or (folder / ".edge-result.json").is_file():
        return {"stage": "Förbereder visningen", "message": meta.get("message", "Skapar förhandsvideo och produktbilder")}
    log = folder / "inference.log"
    if not log.is_file():
        return {"stage": "Förbereder körningen", "message": meta.get("message", "Laddar modeller")}
    # Det räcker med slutet av loggen, även för långa videor.
    with log.open("rb") as handle:
        handle.seek(max(0, log.stat().st_size - 8192))
        lines = handle.read().decode("utf-8", errors="replace").splitlines()
    for line in reversed(lines):
        match = re.fullmatch(r"ViT objekt (\S+): bild (\d+) av (\d+) klar\.", line)
        if match:
            object_id, done, total = match.groups()
            return {"stage": "ViT", "message": f"Objekt #{object_id} — {done} av {total} bilder klara", "completed": int(done), "total": int(total)}
        match = re.fullmatch(r"ViT klassificerar objekt (\S+) med (\d+) bilder\.", line)
        if match:
            object_id, total = match.groups()
            return {"stage": "ViT", "message": f"Objekt #{object_id} — 0 av {total} bilder klara", "completed": 0, "total": int(total)}
        if line.startswith("YOLO och spårning är klara"):
            return {"stage": "ViT", "message": "Bildrutorna är analyserade. Förbereder produktklassificeringen."}
        match = re.fullmatch(r"YOLO och spårning: (\d+) av (\d+) bildrutor\.", line)
        if match:
            done, total = map(int, match.groups())
            return {"stage": "YOLO och spårning", "message": f"{done:,} av {total:,} bildrutor".replace(",", " "), "completed": done, "total": total}
        match = re.match(r"Öppnat klippet: (\d+) bildrutor,", line)
        if match:
            return {"stage": "Laddar YOLO", "message": f"Klippet innehåller {int(match[1]):,} bildrutor".replace(",", " ")}
    return {"stage": "Laddar modeller", "message": meta.get("message", "Förbereder analysen")}
