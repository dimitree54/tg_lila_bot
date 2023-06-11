import os
import tempfile

import openai
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CallbackContext, filters, CommandHandler

from agent import Lila
from utils import ogg_to_mp3, mp3_to_text, text_to_mp3_multi_language


class TelegramBot:
    def __init__(self, token: str, lila: Lila):
        self.application = ApplicationBuilder().token(token=token).build()
        self.application.add_handler(CommandHandler("forget", self.command_handler))
        self.application.add_handler(MessageHandler(filters.VOICE, self.voice_handler))
        self.application.add_handler(MessageHandler(filters.TEXT, self.text_handler))
        self.lila = lila

    def run_polling(self):
        self.application.run_polling()

    @staticmethod
    async def _load_voice_mp3(update: Update, context: CallbackContext, mp3_path: str):
        voice_file = await context.bot.getFile(update.message.voice.file_id)
        with tempfile.NamedTemporaryFile(suffix=".ogg") as ogg_file:
            await voice_file.download_to_drive(ogg_file.name)
            ogg_to_mp3(ogg_file.name, mp3_path)

    async def voice_handler(self, update: Update, context: CallbackContext) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mp3") as mp3_file:
            await self._load_voice_mp3(update, context, mp3_file.name)
            transcript = mp3_to_text(mp3_file.name)
            answer = await self.lila.arun(update.message.from_user.id, transcript)
            if text_to_mp3_multi_language(answer, mp3_file.name) is None:
                await update.message.reply_text(answer)
            else:
                await update.message.reply_voice(voice=mp3_file.name, caption=answer)

    async def text_handler(self, update: Update, context: CallbackContext) -> None:  # noqa
        answer = await self.lila.arun(update.message.from_user.id, update.message.text)
        await update.message.reply_text(answer)

    async def command_handler(self, update: Update, context: CallbackContext) -> None:  # noqa
        if update.message.text == "/forget":
            self.lila.forget(update.message.from_user.id)
            await update.message.reply_text("Chat history has been forgotten.")
        else:
            await update.message.reply_text("Unknown command.")


if __name__ == '__main__':
    load_dotenv()
    openai.api_key = os.environ["OPENAI_API_KEY"]
    TelegramBot(
        token=os.environ["TELEGRAM_TOKEN"],
        lila=Lila(save_path=os.environ["SAVE_PATH"])
    ).run_polling()
