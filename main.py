import argparse
import os
from pathlib import Path

import openai
import yaml
from dotenv import load_dotenv

from agents.helper_agent import HelperAgent
from agents.web_researcher import WebResearcherAgent
from configs.config import Config
from telegram_bot.tg_bot import TelegramBot

parser = argparse.ArgumentParser()
parser.add_argument("--config_name", type=str)
args = parser.parse_args()

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]
configs_path = str(Path(__file__).parent / "configs")
prompts_dir = str(Path(__file__).parent / "prompts")
config = Config.load(os.path.join(configs_path, args.config_name + ".yaml"))
agent_prompts_path = os.path.join(prompts_dir, config.prompts_name + ".yaml")
with open(agent_prompts_path, 'r') as f:
    agent_prompts = yaml.safe_load(f)
web_researcher_prompts_path = os.path.join(prompts_dir, "web_researcher.yaml")
with open(web_researcher_prompts_path, 'r') as f:
    web_researcher_prompts = yaml.safe_load(f)
web_researcher_agent = WebResearcherAgent(web_researcher_prompts)
agent = HelperAgent(
    os.path.join(os.environ["SAVE_PATH"], config.save_dir_name),
    agent_prompts,
    web_researcher_agent
)
TelegramBot(token=os.environ[config.telegram_token_name], agent=agent).run_polling()
