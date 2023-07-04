from typing import Optional

import yaml
from pydantic import BaseModel


class Prompts(BaseModel):
    prefix: str
    important_memory_suffix: Optional[str] = None

    @classmethod
    def load(cls, config_path: str):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)
