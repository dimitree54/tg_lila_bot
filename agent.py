import os.path

from langchain.agents import initialize_agent, AgentType, load_tools, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.memory import FileChatMessageHistory, ConversationSummaryBufferMemory
from langchain.memory.chat_memory import BaseChatMemory

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

    def _load_short_term_memory(self, user_id: int) -> BaseChatMemory:
        chat_memory_path = self._get_chat_memory_path(user_id)
        summary_memory_path = self._get_summary_memory_path(user_id)
        if os.path.isfile(summary_memory_path):
            with open(summary_memory_path, "r") as f:
                summary_memory = f.read()
        else:
            summary_memory = ""
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613", temperature=0)
        return ConversationSummaryBufferMemory(
            llm=llm, max_token_limit=6000,
            memory_key="chat_history", return_messages=True,
            chat_memory=FileChatMessageHistory(chat_memory_path),
            moving_summary_buffer=summary_memory
        )

    def _save_memory(self, user_id: int, memory: ConversationSummaryBufferMemory):
        summary_memory_path = self._get_summary_memory_path(user_id)
        with open(summary_memory_path, "w") as f:
            f.write(memory.moving_summary_buffer)

    def forget(self, user_id: int):
        memory = self._load_short_term_memory(user_id=user_id)
        memory.clear()
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
            return answer
        except Exception as e:
            return str(e)
