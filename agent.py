import json
import os.path
from typing import List

from langchain.agents import initialize_agent, AgentType, load_tools, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory, ChatMessageHistory
from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema import BaseMessage, messages_from_dict, messages_to_dict, BaseChatMessageHistory

PREFIX = """You name is Lila, you are AI-friend of the user.
It is important that user feels you are friend, not his assistant (you are equal in conversation).
You are not trying to help user, unless they ask you to. Just keeping conversation interesting and natural."""


class Lila:
    def __init__(self, save_path: str):
        self.save_path = save_path
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

    def _get_chat_memory_path(self, user_id) -> str:
        return os.path.join(self.save_path, f"{user_id}_chat.json")

    def _get_summary_memory_path(self, user_id) -> str:
        return os.path.join(self.save_path, f"{user_id}_summary.json")

    @staticmethod
    def _load_messages(save_path: str) -> List[BaseMessage]:
        with open(save_path, "r") as f:
            items = json.load(f)
        messages = messages_from_dict(items)
        return messages

    @staticmethod
    def _save_messages(messages: List[BaseMessage], save_path: str):
        messages_dict = messages_to_dict(messages)
        with open(save_path, "w") as f:
            json.dump(messages_dict, f)

    def _load_chat_memory(self, user_id: int) -> ChatMessageHistory:
        chat_memory_path = self._get_chat_memory_path(user_id)
        if os.path.isfile(chat_memory_path):
            messages = self._load_messages(chat_memory_path)
            chat_memory = ChatMessageHistory(messages=messages)
        else:
            chat_memory = ChatMessageHistory()
        return chat_memory

    def _save_chat_memory(self, user_id: int, chat_memory: BaseChatMessageHistory):
        chat_memory_path = self._get_chat_memory_path(user_id)
        messages = chat_memory.messages
        self._save_messages(messages=messages, save_path=chat_memory_path)

    def _load_summary_memory(self, user_id: int) -> str:
        summary_memory_path = self._get_summary_memory_path(user_id)
        if os.path.isfile(summary_memory_path):
            with open(summary_memory_path, "r") as f:
                summary_memory = f.read()
        else:
            summary_memory = ""
        return summary_memory

    def _save_summary_memory(self, user_id: int, summary_memory: str):
        summary_memory_path = self._get_summary_memory_path(user_id)
        with open(summary_memory_path, "w") as f:
            f.write(summary_memory)

    def _load_short_term_memory(self, user_id: int) -> ConversationSummaryBufferMemory:
        chat_memory = self._load_chat_memory(user_id)
        summary_memory = self._load_summary_memory(user_id)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613", temperature=0)
        return ConversationSummaryBufferMemory(
            llm=llm, max_token_limit=6000,
            memory_key="chat_history", return_messages=True,
            chat_memory=chat_memory,
            moving_summary_buffer=summary_memory
        )

    def _save_short_term_memory(self, user_id: int, memory: ConversationSummaryBufferMemory):
        self._save_chat_memory(user_id=user_id, chat_memory=memory.chat_memory)
        self._save_summary_memory(user_id=user_id, summary_memory=memory.moving_summary_buffer)

    def forget(self, user_id: int):
        chat_memory_path = self._get_chat_memory_path(user_id)
        if os.path.isfile(chat_memory_path):
            os.remove(chat_memory_path)
        summary_memory_path = self._get_summary_memory_path(user_id)
        if os.path.isfile(summary_memory_path):
            os.remove(summary_memory_path)

    @staticmethod
    def _initialise_agent(memory: BaseChatMemory) -> AgentExecutor:
        llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        tools = load_tools(["serpapi"], llm=llm)
        return initialize_agent(
            tools=tools,
            llm=llm,
            verbose=True,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            agent_kwargs={
                "system_message": PREFIX,
            }
        )

    async def arun(self, user_id: int, request: str) -> str:
        try:
            memory = self._load_short_term_memory(user_id=user_id)
            agent = self._initialise_agent(memory=memory)
            answer = await agent.arun(input=request)
            self._save_short_term_memory(user_id=user_id, memory=memory)
            return answer
        except Exception as e:
            return str(e)
