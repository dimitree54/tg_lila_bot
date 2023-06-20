import os.path
from datetime import datetime

from langchain import PromptTemplate
from langchain.agents import load_tools, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.memory.chat_memory import BaseChatMemory
from langchain.prompts import MessagesPlaceholder, \
    HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import SystemMessage
from yid_langchain_extensions.agent.simple_agent import SimpleAgent
from yid_langchain_extensions.output_parser.action_parser import ActionParser
from yid_langchain_extensions.output_parser.thoughts_json_parser import Thought
from yid_langchain_extensions.tools.utils import format_tools, format_tool_names, FinalAnswerTool

from chat_end_detector import SmartMemoryCleaner
from memory import SavableSummaryBufferMemoryWithDates

PREFIX = """You name is Lila (it is female name), you are AI-friend of the user.
It is important that user feels you are friend, not his assistant (you are equal in conversation).
You are not trying to help user, unless they ask you to. Just keeping conversation interesting and natural.
Your conversation is happening in the Telegram messenger.
Current date time is {{date}}"""


class Lila:
    def __init__(self, save_path: str):
        self.save_path = save_path
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        self.memory_cleaner = SmartMemoryCleaner()

        self.llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        final_answer_tool = FinalAnswerTool()
        self.tools = load_tools(["serpapi"], llm=self.llm) + [final_answer_tool]
        self.output_parser = ActionParser.from_extra_thoughts([
            Thought(name="thoughts", description="Your thoughts, user will not see them"),
        ])
        self.format_message = PromptTemplate.from_template(
            self.output_parser.get_format_instructions(), template_format="jinja2").format(
            tool_names=format_tool_names(self.tools)
        )

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

    def _initialise_agent(self, memory: BaseChatMemory) -> AgentExecutor:
        system_message = PromptTemplate.from_template(PREFIX, template_format="jinja2").format(
            date=datetime.now().strftime("%Y-%m-%d %H:%M")
        )
        prompt = ChatPromptTemplate.from_messages(
            messages=[
                SystemMessage(content=system_message),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{{input}}", "jinja2"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
                SystemMessage(content=format_tools(self.tools)),
                SystemMessage(content=self.format_message),
            ]
        )
        agent_executor = SimpleAgent.from_llm_and_prompt(
            llm=self.llm,
            prompt=prompt,
            output_parser=self.output_parser,
            stop_sequences=self.output_parser.stop_sequences,
        ).get_executor(tools=self.tools, memory=memory, verbose=True)
        return agent_executor

    async def arun(self, user_id: int, request: str) -> str:
        memory = self._load_short_term_memory(user_id=user_id)
        agent = self._initialise_agent(memory=memory)
        answer = await agent.arun(input=request)
        return answer

    async def after_message(self, user_id: int):
        memory = self._load_short_term_memory(user_id=user_id)
        await self.memory_cleaner.clean(memory)
