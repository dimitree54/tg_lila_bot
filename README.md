# Telegram Bot Lila
Bot is available by [link](https://github.com/dimitree54/tg_lila_bot).

## Setup
1. `apt-get install ffmpeg` (optional, for voice messages)
2. Get Telegram bot token from @BotFather
3. Get OPENAI_API_KEY with GPT-4 access via [link](https://platform.openai.com)
4. Get SERPAPI_API_KEY (for Google search) via [link](https://serpapi.com)
5. Get GOOGLE_APPLICATION_CREDENTIALS.json (optional, to support voice messages) to use with google-cloud-translate package
6. Get YANDEX_TTS_API_KEY (optional, to support russian voice messages) to use speechkit package
7. Create .env file based on .env_template and fill it with your keys
8. Run main.py

## Acknowledgements
Bot created by Dmitrii Rashchenko, [@dimitree54](https://t.me/dimitree54)
Bot is powered by [OpenAI GPT-4 large language model](https://openai.com/gpt-4).
Bot supports voice messages via
- [Google Text-to-Speech](https://github.com/pndurette/gTTS) - vocalise all languages except russian
- [Yandex SpeechKit](https://github.com/TikhonP/yandex-speechkit-lib-python) - vocalise russian language
- [Google Cloud Translate](https://github.com/googleapis/python-translate) - detect text language (to choose correct TTS engine)
- [OpenAI Whisper](https://openai.com/research/whisper) - speech-to-text
Bot supports Google search via [SerpApi](https://serpapi.com)
Bot agents powered by [LangChain](https://python.langchain.com)
