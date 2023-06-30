import os.path
import shutil
from collections import defaultdict
from datetime import datetime
from multiprocessing import Lock
from typing import Optional

from langchain import PromptTemplate, FAISS
from langchain.agents import load_tools, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory.chat_memory import BaseChatMemory
from langchain.prompts import MessagesPlaceholder, \
    HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import SystemMessage, AIMessage
from yid_langchain_extensions.agent.simple_agent import SimpleAgent
from yid_langchain_extensions.output_parser.action_parser import ActionParser
from yid_langchain_extensions.output_parser.thoughts_json_parser import Thought
from yid_langchain_extensions.tools.utils import format_tools, format_tool_names, FinalAnswerTool

from agents.stm_cleaner import ShortTermMemoryCleaner
from agents.stm_savable import SavableSummaryBufferMemoryWithDates, SavableVeryImportantMemory

PREFIX = """You name is Lila (it is female name), you are AI-friend of the user.
It is important that user feels you are friend, not his assistant (you are equal in conversation).
You are not trying to help user, unless they ask you to. Just keeping conversation interesting and natural.
Your conversation is happening in the Telegram messenger.
You are using markdown for answers, so make sure to always escape all special characters.
Current date time is {{date}}"""

SUMMARIZER_PREFIX = """You need to extract info about user from the conversation and update existing info.
Example of information that you need to extract: name, age, sex, date of birth, hobbies, friends' names and so on.
In other words you need to extract any information that will help to make conversation more personal,
 so user feels you are his friend, that you listen to him and care.
 
For reference, today is {date}."""


class Lila:
    def __init__(self, save_path: str):
        self.save_path = save_path
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        self.short_term_memory_cleaner = ShortTermMemoryCleaner()

        self.llm = ChatOpenAI(model_name="gpt-4-0613", temperature=0)
        final_answer_tool = FinalAnswerTool()
        self.tools = load_tools(["serpapi"], llm=self.llm) + [final_answer_tool]
        self.output_parser = ActionParser.from_extra_thoughts([
            Thought(name="thoughts", description="Your thoughts, user will not see them"),
        ])
        self.format_message = PromptTemplate.from_template(
            self.output_parser.get_format_instructions(), template_format="jinja2").format(
            tool_names=format_tool_names(self.tools)
        )
        self.short_term_memory_llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613", temperature=0)
        self.long_term_memory_embeddings = OpenAIEmbeddings()
        self._memory_locks = defaultdict(Lock)
        self._locks_lock = Lock()

    def _get_memory_lock(self, user_id: int) -> Lock:
        with self._locks_lock:
            return self._memory_locks[user_id]  # noqa

    def _get_user_dir(self, user_id: int) -> str:
        user_dir = os.path.join(self.save_path, str(user_id))
        if not os.path.isdir(user_dir):
            os.makedirs(user_dir)
        return user_dir

    def _get_user_ltm_path(self, user_id: int) -> str:
        return os.path.join(self._get_user_dir(user_id=user_id), "ltm")

    def _load_short_term_memory(self, user_id: int) -> SavableSummaryBufferMemoryWithDates:
        with self._get_memory_lock(user_id=user_id):
            return SavableSummaryBufferMemoryWithDates.load(
                llm=self.short_term_memory_llm, max_token_limit=6000,
                memory_key="chat_history", return_messages=True,
                save_path=self._get_user_dir(user_id=user_id)
            )

    def _load_memory_about_user(self, user_id: int) -> SavableVeryImportantMemory:
        with self._get_memory_lock(user_id=user_id):
            messages = [
                SystemMessage(content=SUMMARIZER_PREFIX.format(date=self._now())),
                HumanMessagePromptTemplate.from_template("Existing info about user:\n{{summary}}", "jinja2"),
                HumanMessagePromptTemplate.from_template("New lines of conversation:\n{{new_lines}}", "jinja2"),
                SystemMessage(content="You have to answer with updated info about user and nothing else."),
            ]
            return SavableVeryImportantMemory.load(
                llm=self.short_term_memory_llm,
                save_path=self._get_user_dir(user_id=user_id),
                summarizer_prompt=ChatPromptTemplate.from_messages(messages=messages),
            )

    def _load_long_term_memory(self, user_id: int) -> Optional[FAISS]:
        with self._get_memory_lock(user_id=user_id):
            ltm_path = self._get_user_ltm_path(user_id=user_id)
            if os.path.exists(ltm_path):
                return FAISS.load_local(ltm_path, self.long_term_memory_embeddings)
            return None

    def _create_long_term_memory(self, first_memory: str) -> FAISS:
        return FAISS.from_texts([first_memory], self.long_term_memory_embeddings,
                                metadatas=[{"date": datetime.now().isoformat()}])

    def _add_to_long_term_memory(self, user_id: int, new_long_term_memory: str):
        long_term_memory = self._load_long_term_memory(user_id=user_id)
        with self._get_memory_lock(user_id=user_id):
            if long_term_memory:
                long_term_memory.add_texts([new_long_term_memory], metadatas=[{"date": datetime.now().isoformat()}])
            else:
                long_term_memory = self._create_long_term_memory(first_memory=new_long_term_memory)
            long_term_memory.save_local(self._get_user_ltm_path(user_id=user_id))

    def forget(self, user_id: int):
        short_term_memory = self._load_short_term_memory(user_id=user_id)
        memory_about_user = self._load_memory_about_user(user_id=user_id)
        with self._get_memory_lock(user_id=user_id):
            short_term_memory.clear()
            memory_about_user.clear()
            ltm_path = self._get_user_ltm_path(user_id=user_id)
            if os.path.exists(ltm_path):
                shutil.rmtree(ltm_path)
        print(f"Memory about user {user_id} forgotten")

    def _get_relevant_ltm(
            self, user_id: int, short_term_memory: BaseChatMemory, long_term_memory: Optional[FAISS]) -> Optional[str]:
        if long_term_memory is None:
            return None
        with self._get_memory_lock(user_id=user_id):
            short_term_memory.return_messages = False
            short_term_context = short_term_memory.load_memory_variables({})["chat_history"]
            short_term_memory.return_messages = True
            relevant_document = long_term_memory.similarity_search(short_term_context, k=1)[0]
            date = datetime.fromisoformat(relevant_document.metadata["date"]).strftime('%Y-%m-%d')
            thought = f"Thought (user does not see it):\n" \
                      f"Hm, that reminds me another conversation I had {date} with user:\n" \
                      f"{relevant_document.page_content}"
            return thought

    def _get_relevant_memory_about_user(self, user_id: int, memory_about_user: SavableVeryImportantMemory) -> str:
        relevant_memory_about_user = f"Thought (user does not see it):\n" \
                                     f"What I know about user so far:\n"
        with self._get_memory_lock(user_id=user_id):
            if memory_about_user.buffer == "":
                relevant_memory_about_user += "Nothing"
            else:
                relevant_memory_about_user += memory_about_user.buffer
        return relevant_memory_about_user

    @staticmethod
    def _now() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M")

    def _initialise_agent(
            self, user_id: int, short_term_memory: BaseChatMemory, memory_about_user: SavableVeryImportantMemory,
            long_term_memory: Optional[FAISS]) -> AgentExecutor:
        system_message = PromptTemplate.from_template(PREFIX, template_format="jinja2").format(
            date=self._now()
        )
        messages = [
            SystemMessage(content=system_message),
            AIMessage(content=self._get_relevant_memory_about_user(user_id, memory_about_user)),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{{input}}", "jinja2"),
        ]
        if (relevant_ltm := self._get_relevant_ltm(user_id, short_term_memory, long_term_memory)) is not None:
            messages.append(AIMessage(content=relevant_ltm))
        messages.extend([
            MessagesPlaceholder(variable_name="agent_scratchpad"),
            SystemMessage(content=format_tools(self.tools)),
            SystemMessage(content=self.format_message),
        ])
        prompt = ChatPromptTemplate.from_messages(messages=messages)
        agent_executor = SimpleAgent.from_llm_and_prompt(
            llm=self.llm,
            prompt=prompt,
            output_parser=self.output_parser,
            stop_sequences=self.output_parser.stop_sequences,
        ).get_executor(tools=self.tools, memory=short_term_memory, verbose=True)
        return agent_executor

    async def arun(self, user_id: int, request: str) -> str:
        short_term_memory = self._load_short_term_memory(user_id=user_id)
        memory_about_user = self._load_memory_about_user(user_id=user_id)
        long_term_memory = self._load_long_term_memory(user_id=user_id)
        agent = self._initialise_agent(user_id, short_term_memory, memory_about_user, long_term_memory)
        answer = await agent.arun(input=request)
        return answer

    @staticmethod
    def _clear_short_term_memory(memory: BaseChatMemory):
        last_request = memory.chat_memory.messages[-2].content
        last_answer = memory.chat_memory.messages[-1].content
        memory.clear()
        memory.save_context({"input": last_request}, {"output": last_answer})

    async def after_message(self, user_id: int):
        short_term_memory = self._load_short_term_memory(user_id=user_id)
        memory_about_user = self._load_memory_about_user(user_id=user_id)
        new_long_term_memory = await self.short_term_memory_cleaner.compress(short_term_memory)
        if new_long_term_memory is not None:
            print(f"Adding to long term memory of user {user_id}: {new_long_term_memory}")
            self._add_to_long_term_memory(user_id=user_id, new_long_term_memory=new_long_term_memory)
            memory_about_user.update(short_term_memory.load_memory_variables({})["chat_history"])
            self._clear_short_term_memory(memory=short_term_memory)
