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


def get_important_info_thought(important_info_description) -> Thought:
    return Thought(
        name="updated_important_info",
        description="(Optional) If based on chat important info may be updated,"
                    " include that field in output json. "
                    "It should include updated important info, including everything you known so fat plus new facts."
                    f"It can be used only together with final_answer action. {important_info_description}"
    )
