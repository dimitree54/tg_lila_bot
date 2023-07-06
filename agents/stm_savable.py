import json
import os
from datetime import datetime
from typing import List, Dict, Any, Callable

from langchain.memory import ConversationSummaryBufferMemory, ChatMessageHistory
from langchain.schema import BaseMessage, messages_from_dict, messages_to_dict


class ChatMessageHistoryWithDates(ChatMessageHistory):
    def add_message(self, message: BaseMessage) -> None:
        message.additional_kwargs["timestamp"] = datetime.now().isoformat()
        super().add_message(message)


def _load_summary_memory(summary_memory_path: str) -> str:
    if os.path.isfile(summary_memory_path):
        with open(summary_memory_path, "r") as f:
            summary_memory = f.read()
    else:
        summary_memory = ""
    return summary_memory


def _load_messages(save_path: str) -> List[BaseMessage]:
    with open(save_path, "r") as f:
        items = json.load(f)
    messages = messages_from_dict(items)
    return messages


def _load_chat_memory_with_dates(chat_memory_path: str) -> ChatMessageHistoryWithDates:
    if os.path.isfile(chat_memory_path):
        messages = _load_messages(chat_memory_path)
        chat_memory = ChatMessageHistoryWithDates(messages=messages)
    else:
        chat_memory = ChatMessageHistoryWithDates()
    return chat_memory


class SavableSummaryBufferMemoryWithDates(ConversationSummaryBufferMemory):
    save_path: str
    chat_memory_file_name: str
    summary_memory_file_name: str
    output_preprocessor: Callable[[Dict], Dict] = lambda x: x

    @classmethod
    def load(cls, save_path: str, **kwargs):
        chat_memory_file_name = kwargs.pop("chat_memory_file_name", "chat_memory.json")
        summary_memory_file_name = kwargs.pop("summary_memory_file_name", "summary_memory.txt")
        chat_memory_path = os.path.join(save_path, chat_memory_file_name)
        summary_memory_path = os.path.join(save_path, summary_memory_file_name)
        chat_memory = _load_chat_memory_with_dates(chat_memory_path)
        summary_memory = _load_summary_memory(summary_memory_path)
        return cls(
            save_path=save_path,
            chat_memory_file_name=chat_memory_file_name,
            summary_memory_file_name=summary_memory_file_name,
            chat_memory=chat_memory, moving_summary_buffer=summary_memory, **kwargs
        )

    @staticmethod
    def _save_messages(messages: List[BaseMessage], save_path: str):
        messages_dict = messages_to_dict(messages)
        with open(save_path, "w") as f:
            json.dump(messages_dict, f)

    def _save_chat_memory(self):
        chat_memory_path = os.path.join(self.save_path, self.chat_memory_file_name)
        messages = self.chat_memory.messages
        self._save_messages(messages, chat_memory_path)

    def _save_summary_memory(self):
        summary_memory_path = os.path.join(self.save_path, self.summary_memory_file_name)
        with open(summary_memory_path, "w") as f:
            f.write(self.moving_summary_buffer)

    def save(self):
        self._save_chat_memory()
        self._save_summary_memory()

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        super().save_context(inputs, self.output_preprocessor(outputs))
        self.save()

    def clear(self) -> None:
        super().clear()
        self.save()
