import argparse
import os
from pathlib import Path

import openai
from dotenv import load_dotenv

from agents.friend_lila import HelperAgent
from configs.config import Config
from prompts.prompts import Prompts
from telegram_bot.tg_bot import TelegramBot

parser = argparse.ArgumentParser()
parser.add_argument("--config_name", type=str)
args = parser.parse_args()

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]
configs_path = str(Path(__file__).parent / "configs")
prompts_dir = str(Path(__file__).parent / "prompts")
config = Config.load(os.path.join(configs_path, args.config_name + ".yaml"))
prompts = Prompts.load(os.path.join(prompts_dir, config.prompts_name + ".yaml"))
agent = HelperAgent(
    os.path.join(os.environ["SAVE_PATH"], config.save_dir_name),
    prompts
)
TelegramBot(token=os.environ[config.telegram_token_name], agent=agent).run_polling()
