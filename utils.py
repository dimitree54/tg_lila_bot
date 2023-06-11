from typing import Optional

import openai
from gtts import gTTS
import librosa
import soundfile as sf
from pydub import AudioSegment

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


def text_to_mp3_multi_language(text: str, mp3_path: str) -> Optional[str]:
    language = LanguageDetector().detect(text)
    if language == TTSLanguage.OTHER:
        return None
    text_to_mp3(text, mp3_path, language)
    if language == TTSLanguage.RUSSIAN:
        speed_up_audio(mp3_path, mp3_path, 1.5)
    return mp3_path


def speed_up_audio(input_path: str, output_path: str, speed: float):
    time_series, sample_rate = librosa.load(input_path)
    time_series_fast = librosa.effects.time_stretch(time_series, rate=speed)
    sf.write(output_path, time_series_fast, samplerate=int(sample_rate))
