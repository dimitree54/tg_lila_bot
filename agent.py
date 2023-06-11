from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

load_dotenv()

agent = initialize_agent(
    tools=[],
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
    verbose=True,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
)
