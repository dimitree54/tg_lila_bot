from datetime import datetime


def format_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")
