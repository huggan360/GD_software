# Visningen använder verktygets egen status, inte C++-programmets loggar.
def read_progress(meta):
    if meta["status"] != "running":
        return None
    stage = "Förbereder visningen" if meta.get("progress", 0) >= 90 else "Kör modellerna"
    return {"stage": stage, "message": meta.get("message", stage)}
