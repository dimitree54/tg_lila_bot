import os
from unittest import TestCase

from dotenv import load_dotenv

from language_detector import LanguageDetector, TTSLanguage


class TestText(TestCase):
    def setUp(self) -> None:
        load_dotenv()
        with open(os.environ["GOOGLE_APPLICATION_CREDENTIALS"], 'r') as file:
            print(file.read())

    def test_language_detector(self):
        detector = LanguageDetector()
        test_text = "Привет, меня зовут Лиила."
        language = detector.detect(test_text)
        self.assertEquals(language, TTSLanguage.RUSSIAN)
        test_text = "Hi, my name is Liila."
        language = detector.detect(test_text)
        self.assertEquals(language, TTSLanguage.ENGLISH)
        test_text = "Laba diena, mano vardas Liila"
        language = detector.detect(test_text)
        self.assertEquals(language, TTSLanguage.OTHER)
