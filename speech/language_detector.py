from enum import Enum

from google.cloud import translate_v2 as translate


class TTSLanguage(Enum):
    AFRIKAANS = "af"
    ARABIC = "ar"
    BULGARIAN = "bg"
    BENGALI = "bn"
    BOSNIAN = "bs"
    CATALAN = "ca"
    CZECH = "cs"
    DANISH = "da"
    GERMAN = "de"
    GREEK = "el"
    ENGLISH = "en"
    SPANISH = "es"
    ESTONIAN = "et"
    FINNISH = "fi"
    FRENCH = "fr"
    GUJARATI = "gu"
    HINDI = "hi"
    CROATIAN = "hr"
    HUNGARIAN = "hu"
    INDONESIAN = "id"
    ICELANDIC = "is"
    ITALIAN = "it"
    HEBREW = "iw"
    JAPANESE = "ja"
    JAVANESE = "jw"
    KHMER = "km"
    KANNADA = "kn"
    KOREAN = "ko"
    LATIN = "la"
    LATVIAN = "lv"
    MALAYALAM = "ml"
    MARATHI = "mr"
    MALAY = "ms"
    MYANMAR = "my"
    NEPALI = "ne"
    DUTCH = "nl"
    NORWEGIAN = "no"
    POLISH = "pl"
    PORTUGUESE = "pt"
    ROMANIAN = "ro"
    RUSSIAN = "ru"
    SINHALA = "si"
    SLOVAK = "sk"
    ALBANIAN = "sq"
    SERBIAN = "sr"
    SUNDANESE = "su"
    SWEDISH = "sv"
    SWAHILI = "sw"
    TAMIL = "ta"
    TELUGU = "te"
    THAI = "th"
    FILIPINO = "tl"
    TURKISH = "tr"
    UKRAINIAN = "uk"
    URDU = "ur"
    VIETNAMESE = "vi"
    CHINESE_SIMPLIFIED = "zh-cn"
    CHINESE_TRADITIONAL = "zh-tw"
    CHINESE = "zh"
    OTHER = "other"


class LanguageDetector:
    def __init__(self):
        self.translate_client = translate.Client()

    def detect(self, text: str) -> TTSLanguage:
        try:
            language_string = self.translate_client.detect_language(text)["language"]
            language = TTSLanguage(language_string)
        except ValueError:
            language = TTSLanguage.OTHER
        return language
