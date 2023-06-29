from copy import deepcopy
from datetime import datetime
from typing import List, Optional

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain.schema import BaseMessage
from yid_langchain_extensions.output_parser.class_parser import Class, ClassParser, get_classes_description, \
    get_classes_summary
from yid_langchain_extensions.output_parser.thoughts_json_parser import Thought

SUFFIX = """There were a conversation between Human and AI assistant:

{{conversation_summary}}

Last message of which were from AI assistant:
------
{{last_message}}
------

And after {{delay_hours}} hours Human sends a new message:
------
{{new_message}}
------

Your task is to classify that new message into one of the following categories:

{{classes_description}}

Consider messages content and delay between messages"""


class SmartMemoryCleaner:
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613", temperature=0)
        self.classes = [
            Class(
                name="CONTINUE",
                description="the new message is a response to the last message"
            ),
            Class(
                name="NEW",
                description="the new message is a start of a new topic/conversation, not related to the previous one"
            ),
        ]
        self.output_parser = ClassParser.from_extra_thoughts(
            extra_thoughts=[
                Thought(
                    name="Evidences for CONTINUE",
                    description="Evidences that new message is a response to the last message"
                ),
                Thought(
                    name="Evidences for NEW",
                    description="Evidences that new message is about a new topic"
                ),
                Thought(
                    name="Thoughts",
                    description="Your thoughts of what class to choose"
                )
            ]
        )
        self.chat_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(SUFFIX, "jinja2"),
                SystemMessagePromptTemplate.from_template(self.output_parser.get_format_instructions(), "jinja2"),
            ]
        )

    @staticmethod
    def _get_hours_after_message(message: BaseMessage) -> int:
        time1 = datetime.fromisoformat(message.additional_kwargs["timestamp"])
        time2 = datetime.now()
        return int(round((time2 - time1).total_seconds() / 3600))

    def _format_messages(
            self, summary: str, delay_hours: int, last_message: str, new_message: str) -> List[BaseMessage]:
        prompt_messages = self.chat_template.format_messages(
            conversation_summary=summary,
            new_message=new_message,
            last_message=last_message,
            delay_hours=delay_hours,
            classes_description=get_classes_description(self.classes),
            classes_summary=get_classes_summary(self.classes)
        )
        return prompt_messages

    @staticmethod
    def get_previous_conversation_summary(memory: ConversationSummaryBufferMemory) -> str:
        previous_chat_memory = deepcopy(memory)
        previous_chat_memory.chat_memory.messages.pop()
        previous_chat_memory.chat_memory.messages.pop()
        previous_chat_memory.max_token_limit = 3  # empty chat takes 3 tokens
        previous_chat_memory.prune()
        return previous_chat_memory.moving_summary_buffer

    async def _is_new_conversation(
            self, previous_conversation_summary: str, last_message: BaseMessage, new_message: str) -> bool:
        delay_hours = self._get_hours_after_message(last_message)
        if delay_hours > 6:
            return True
        messages = self._format_messages(
            summary=previous_conversation_summary,
            delay_hours=delay_hours,
            last_message=last_message.content,
            new_message=new_message
        )
        prediction = await self.llm.apredict_messages(messages, stop=self.output_parser.stop_sequences)
        class_index = self.output_parser.parse(prediction.content)
        return class_index == 1

    async def compress_stm(self, memory: ConversationSummaryBufferMemory) -> Optional[str]:
        if len(memory.chat_memory.messages) < 2:
            return None
        last_message = memory.chat_memory.messages[-3]
        last_request = memory.chat_memory.messages[-2].content
        last_answer = memory.chat_memory.messages[-1].content
        previous_conversation_summary = self.get_previous_conversation_summary(memory)
        is_new_conversation = await self._is_new_conversation(previous_conversation_summary, last_message, last_request)
        if is_new_conversation:
            memory.clear()
            memory.save_context({"input": last_request}, {"output": last_answer})
            return previous_conversation_summary
        return None
