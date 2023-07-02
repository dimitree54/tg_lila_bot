import os.path
import shutil
from collections import defaultdict
from datetime import datetime
from multiprocessing import Lock
from typing import Optional

from langchain import PromptTemplate, FAISS
from langchain.agents import AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory.chat_memory import BaseChatMemory
from langchain.prompts import MessagesPlaceholder, \
    HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate
from langchain.schema import SystemMessage, AIMessage
from yid_langchain_extensions.agent.simple_agent import SimpleAgent
from yid_langchain_extensions.output_parser.action_parser import ActionParser
from yid_langchain_extensions.output_parser.thoughts_json_parser import Thought
from yid_langchain_extensions.tools.utils import format_tools, format_tool_names, FinalAnswerTool

from agents.stm_cleaner import ShortTermMemoryCleaner
from agents.stm_savable import SavableSummaryBufferMemoryWithDates, SavableVeryImportantMemory
from agents.tools import WebSearchTool, AskPageTool

PREFIX = """You name is Lila (it is female name), you are AI-friend of the user.
It is important that user feels you are friend, not his assistant (you are equal in conversation).
You are not trying to help user, unless they ask you to. Just keeping conversation interesting and natural.
Your conversation is happening in the Telegram messenger.
You are using markdown for answers, so make sure to always escape all special characters.

You will be provided with web tools: web_search and ask_url.
web_search is like google, a tool to get relevant links with a short snippet from the page.
It is fast and cheap, but does not provide rich information
ask_url is like visiting use it to get answer based on full page content.
It is slow and expensive, but provides rich information based on full page content.
Use following pipeline to answer web-based questions:
1. Use web_search to get relevant links
2. Estimate if link is useful based on its snippet.
3. If you are not sure that found useful link, refine your web_search query and go to step 1.
4. If you are sure that found useful link, use ask_url to get answer based on full page content.

Prefer using ask_url to get more informative answer, rather than answering based on web_search snippets.
Include markdown-formatted links that you found useful in your answer.

Current date time is {{date}}"""

SUMMARIZER_SUFFIX = """It was a conversation between AI and human.
You need to extract any information about the user that will help to make conversation with him more personal,
 so user feels that AI are his friend, that AI listen to him and care.
But do not include conversation details, its topic, what were discussed, etc.
Do not include any information that is temporary relevant, for example, plans for the day.
Only persistent information about user as person that does not change often.

Using that extracted information, update what you already know about the user with new information.

For reference, today is {date}.

Example of relevant information about user:
User name is Poul, he lives in Argentina, he is 25 years old, he likes to play football, he has a dog named Rex.
He speak Spanish and want AI to speak Spanish too. He does not like too much questions from AI.
His birthday is 25th of December.
He has a friend named John, he is 30 years old, they play football together for 3 years.

Example of irrelevant information about user:
User asked AI for recipes of pizza, AI answered with recipe of pizza, user said "thanks".

Information about user that you already know:
{summary}

Updated information about user:
"""


class Lila:
    def __init__(self, save_path: str):
        self.save_path = save_path
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        self.short_term_memory_cleaner = ShortTermMemoryCleaner()

        self.smart_llm = ChatOpenAI(model_name="gpt-4-0613", temperature=0)
        self.fast_llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613", temperature=0)

        final_answer_tool = FinalAnswerTool()
        web_search_tool = WebSearchTool()
        ask_url_tool = AskPageTool(llm=self.smart_llm)

        self.tools = [final_answer_tool, web_search_tool, ask_url_tool]
        self.output_parser = ActionParser.from_extra_thoughts([
            Thought(name="thoughts", description="Your thoughts about what action to take, user will not see them"),
            Thought(name="self_criticism", description="Your self-criticism about what you said, user will not see it"),
        ])
        self.format_message = PromptTemplate.from_template(
            self.output_parser.get_format_instructions(), template_format="jinja2").format(
            tool_names=format_tool_names(self.tools)
        )
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
                llm=self.fast_llm, max_token_limit=6000,
                memory_key="chat_history", return_messages=True,
                save_path=self._get_user_dir(user_id=user_id)
            )

    def _load_memory_about_user(self, user_id: int) -> SavableVeryImportantMemory:
        with self._get_memory_lock(user_id=user_id):
            suffix_template = PromptTemplate.from_template(SUMMARIZER_SUFFIX).partial(date=self._now())
            messages = [
                HumanMessagePromptTemplate.from_template("Conversation between AI and human:\n{{new_lines}}", "jinja2"),
                SystemMessagePromptTemplate(prompt=suffix_template),
            ]
            return SavableVeryImportantMemory.load(
                llm=self.fast_llm,
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
            thought = "Thought (user does not see it):\n" \
                      f"Hm, that reminds me another conversation I had {date} with user:\n" \
                      f"{relevant_document.page_content}"
            return thought

    def _get_relevant_memory_about_user(self, user_id: int, memory_about_user: SavableVeryImportantMemory) -> str:
        relevant_memory_about_user = "Thought (user does not see it):\n" \
                                     "What I know about user so far:\n"
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
            llm=self.smart_llm,
            prompt=prompt,
            output_parser=self.output_parser,
            stop_sequences=self.output_parser.stop_sequences,
        ).get_executor(tools=self.tools, memory=short_term_memory, verbose=True)
        return agent_executor

    async def arun(self, user_id: int, request: str) -> str:
        print("Starting on message")
        try:
            short_term_memory = self._load_short_term_memory(user_id=user_id)
            memory_about_user = self._load_memory_about_user(user_id=user_id)
            long_term_memory = self._load_long_term_memory(user_id=user_id)
            agent = self._initialise_agent(user_id, short_term_memory, memory_about_user, long_term_memory)
            answer = await agent.arun(input=request)
            print("Finished on message")
            return answer
        except Exception as e:
            return f"Error in telegram bot: {e}. Report it to developer."

    @staticmethod
    def _clear_short_term_memory(memory: BaseChatMemory):
        last_request = memory.chat_memory.messages[-2].content
        last_answer = memory.chat_memory.messages[-1].content
        memory.clear()
        memory.save_context({"input": last_request}, {"output": last_answer})

    async def after_message(self, user_id: int):
        print("Starting after message")
        short_term_memory = self._load_short_term_memory(user_id=user_id)
        memory_about_user = self._load_memory_about_user(user_id=user_id)
        new_long_term_memory = await self.short_term_memory_cleaner.compress(short_term_memory)
        if new_long_term_memory is not None:
            print(f"Adding to long term memory of user {user_id}: {new_long_term_memory}")
            self._add_to_long_term_memory(user_id=user_id, new_long_term_memory=new_long_term_memory)
            memory_about_user.update(short_term_memory.load_memory_variables({})["chat_history"])
            self._clear_short_term_memory(memory=short_term_memory)
        print("Finished after message")
