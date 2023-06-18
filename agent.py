import os.path

from langchain.agents import initialize_agent, AgentType, load_tools, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.memory.chat_memory import BaseChatMemory

from chat_end_detector import ChatEndDetector
from memory import SavableSummaryBufferMemoryWithDates

PREFIX = """You name is Lila (it is female name), you are AI-friend of the user.
It is important that user feels you are friend, not his assistant (you are equal in conversation).
You are not trying to help user, unless they ask you to. Just keeping conversation interesting and natural."""


class Lila:
    def __init__(self, save_path: str):
        self.save_path = save_path
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        self.chat_end_detector = ChatEndDetector()

    def _load_short_term_memory(self, user_id: int) -> SavableSummaryBufferMemoryWithDates:
        save_path = os.path.join(self.save_path, str(user_id))
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613", temperature=0)
        return SavableSummaryBufferMemoryWithDates.load(
            llm=llm, max_token_limit=6000,
            memory_key="chat_history", return_messages=True,
            save_path=save_path
        )

    def forget(self, user_id: int):
        memory = self._load_short_term_memory(user_id=user_id)
        memory.clear()

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
            is_new_chat = self.chat_end_detector.is_new_conversation(memory, request)
            if is_new_chat:
                memory.clear()
            agent = self._initialise_agent(memory=memory)
            answer = await agent.arun(input=request)
            return answer
        except Exception as e:
            return str(e)
