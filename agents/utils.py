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


def get_conversation_summary_thought() -> Thought:
    return Thought(
        name="updated_conversation_summary",
        description="Summarise, what current conversation is about. "
                    "Consider existing conversation summary, and add new facts to it. "
                    "If nothing new, just repeat existing conversation summary. "
                    "Include all important topics discussed in conversation and what conclusions were made. "
                    "Always use it together with final_answer action."
    )


def get_end_detection_thought() -> Thought:
    return Thought(
        name="new_topic_started",
        type="bool",
        description="If it seems that user have started discussing another topic,"
                    " different from the current one,"
                    " set that field to True. "
                    "Also consider user's greeting as a new topic start. "
                    "Also consider significant time gap after last user's message as a new topic start. "
                    "Always use it together with final_answer action."
    )


def get_important_info_thought(important_info_description: str) -> Thought:
    return Thought(
        name="updated_important_info",
        description="(Optional) If based on chat important info may be updated,"
                    " include that field in output json. "
                    "It should include updated important info, including everything you known so far plus new facts. "
                    "If fact is temporary relevant (for example about today or current week),"
                    " include it with time period when it is relevant, so you can remove it when no longer relevant. "
                    f"It can be used only together with final_answer action. {important_info_description}"
    )
