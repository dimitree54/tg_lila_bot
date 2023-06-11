import os

import openai
from dotenv import load_dotenv

from agent import Lila
from tg_bot import TelegramBot

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]
TelegramBot(
    token=os.environ["TELEGRAM_TOKEN"],
    lila=Lila(save_path=os.environ["SAVE_PATH"])
).run_polling()
