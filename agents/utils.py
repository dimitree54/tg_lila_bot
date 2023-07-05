from datetime import datetime

from yid_langchain_extensions.output_parser.thoughts_json_parser import Thought


def format_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def get_thought_thought() -> Thought:
    return Thought(
        name="thoughts",
        description="Your thoughts about what action to take now and planning further steps.")


def get_self_criticism_thought() -> Thought:
    return Thought(
        name="self_criticism",
        description="Your self-criticism about what you said, considering alternative options.")
