import os.path

from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory


PREFIX = """You name is Lila, you are AI-friend of the user.
It is important that user feels you are friend, not his assistant (you are equal in conversation).
You are not trying to help user, unless they ask you to. Just keeping conversation interesting and natural."""


class Lila:
    def __init__(self, save_path: str):
        self.save_path = save_path
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

    def _load_memory(self, user_id: int) -> ConversationBufferMemory:
        memory_path = os.path.join(self.save_path, f"{user_id}.json")
        return ConversationBufferMemory(
            memory_key="chat_history", return_messages=True,
            chat_memory=FileChatMessageHistory(memory_path)
        )

    def forget(self, user_id: int):
        memory = self._load_memory(user_id=user_id)
        memory.clear()

    @staticmethod
    def _initialise_agent(memory: ConversationBufferMemory):
        return initialize_agent(
            tools=[],
            llm=ChatOpenAI(model_name="gpt-4", temperature=0),
            verbose=True,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            agent_kwargs={
                "system_message": PREFIX,
            }
        )

    async def arun(self, user_id: int, request: str) -> str:
        try:
            memory = self._load_memory(user_id=user_id)
            agent = self._initialise_agent(memory=memory)
            answer = await agent.arun(input=request)
            return answer
        except Exception as e:
            return str(e)
