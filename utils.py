import os
import tempfile
from typing import Optional

import openai
from gtts import gTTS
from pydub import AudioSegment
from speechkit import Session, SpeechSynthesis

from language_detector import TTSLanguage, LanguageDetector


def ogg_to_mp3(ogg_path, mp3_path):
    audio = AudioSegment.from_ogg(ogg_path)
    audio.export(mp3_path, format="mp3")


def mp3_to_text(mp3_path: str) -> str:
    with open(mp3_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe(model="whisper-1", file=audio_file)
        return transcript["text"]


def text_to_mp3(text: str, mp3_path: str, language: TTSLanguage):
    tts = gTTS(text, lang=language.value, slow=False)
    tts.save(mp3_path)


def text_to_ogg_yandex(text: str, wav_path: str):
    session = Session.from_api_key(os.environ["YANDEX_TTS_API_KEY"])
    SpeechSynthesis(session).synthesize(
        wav_path, text=text, voice='alena', emotion='good'
    )


def text_to_mp3_multi_language(text: str, mp3_path: str) -> Optional[str]:
    language = LanguageDetector().detect(text)
    if language == TTSLanguage.OTHER:
        return None
    text_to_mp3(text, mp3_path, language)
    if language == TTSLanguage.RUSSIAN:
        with tempfile.NamedTemporaryFile(suffix=".ogg") as ogg_file:
            text_to_ogg_yandex(text, ogg_file.name)
            ogg_to_mp3(ogg_file.name, mp3_path)
    return mp3_path
