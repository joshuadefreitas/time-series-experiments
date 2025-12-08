import json
from pathlib import Path
from typing import Any, Dict


def save_json(data: Dict[str, Any], path: str | Path) -> Path:
    """
    Save a dictionary to JSON with UTF-8 and nice indentation.
    Creates parent directories if needed.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return out_path
