import os
from pathlib import Path


def ai_data_root() -> Path:
    return Path(os.environ["AI_DATA_ROOT"])


def artefact_path(*parts: str) -> Path:
    return ai_data_root().joinpath(*parts)
