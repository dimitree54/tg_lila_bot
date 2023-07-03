import yaml
from pydantic import BaseModel


class Config(BaseModel):
    telegram_token_name: str
    save_dir_name: str
    prompts_name: str

    @classmethod
    def load(cls, config_path: str):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)
