import shutil
import tempfile
from typing import List
from unittest import TestCase, IsolatedAsyncioTestCase

import tiktoken
from dotenv import load_dotenv
from langchain.base_language import BaseLanguageModel
from langchain.chat_models import ChatOpenAI
from langchain.llms import FakeListLLM
from langchain.memory.chat_memory import BaseChatMemory

from agents.stm_cleaner import ShortTermMemoryCleaner
from agents.stm_savable import SavableSummaryBufferMemoryWithDates


def add_test_messages(memory: BaseChatMemory):
    memory.save_context({"input": "hi"}, {"output": "whats up"})
    memory.save_context({"input": "not much you"}, {"output": "not much"})
    memory.save_context({"input": "bye"}, {"output": "see you"})


def load_memory(llm: BaseLanguageModel, save_path: str, token_limit: int = 6000) -> SavableSummaryBufferMemoryWithDates:
    return SavableSummaryBufferMemoryWithDates.load(
        llm=llm, max_token_limit=token_limit,
        memory_key="chat_history", return_messages=True,
        save_path=save_path
    )


class FakeListLLMTiktoken(FakeListLLM):
    # Because of some bug with tokenizers, FakeListLLM too slow
    #  (it downloads the tokenizer model every call)
    #  so for testing we use tiktoken instead
    def get_token_ids(self, text: str) -> List[int]:
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return enc.encode(text)


class TestMemory(TestCase):
    def setUp(self) -> None:
        load_dotenv()
        self.save_path = tempfile.mkdtemp()
        self.test_user_id = 0
        self.fake_summary = "blah blah blah"
        self.llm = FakeListLLMTiktoken(responses=[self.fake_summary])

    def test_memory(self):
        memory = load_memory(self.llm, self.save_path)
        add_test_messages(memory)
        self.assertEqual(memory.load_memory_variables({})["chat_history"][0].content, "hi")

    def test_memory_savable(self):
        memory = load_memory(self.llm, self.save_path)
        add_test_messages(memory)
        memory = load_memory(self.llm, self.save_path)
        self.assertEqual(memory.load_memory_variables({})["chat_history"][0].content, "hi")

    def test_memory_clearable(self):
        memory = load_memory(self.llm, self.save_path)
        add_test_messages(memory)
        memory.clear()
        self.assertEqual(memory.load_memory_variables({})["chat_history"], [])
        self.assertEqual(memory.moving_summary_buffer, "")
        self.assertEqual(memory.chat_memory.messages, [])
        memory = load_memory(self.llm, self.save_path)
        self.assertEqual(memory.load_memory_variables({})["chat_history"], [])
        self.assertEqual(memory.moving_summary_buffer, "")
        self.assertEqual(memory.chat_memory.messages, [])

    def test_memory_summarization(self):
        memory = load_memory(self.llm, self.save_path, 20)
        add_test_messages(memory)
        self.assertEqual(memory.load_memory_variables({})["chat_history"][0].content, self.fake_summary)

    def test_memory_summarization_savable(self):
        memory = load_memory(self.llm, self.save_path, 20)
        add_test_messages(memory)
        memory = load_memory(self.llm, self.save_path, 20)
        self.assertEqual(memory.load_memory_variables({})["chat_history"][0].content, self.fake_summary)

    def test_date(self):
        memory = load_memory(self.llm, self.save_path)
        add_test_messages(memory)
        self.assertTrue("timestamp" in memory.load_memory_variables({})["chat_history"][0].additional_kwargs)

    def test_date_savable(self):
        memory = load_memory(self.llm, self.save_path)
        add_test_messages(memory)
        date = memory.load_memory_variables({})["chat_history"][0].additional_kwargs["timestamp"]
        memory = load_memory(self.llm, self.save_path)
        self.assertEqual(memory.load_memory_variables({})["chat_history"][0].additional_kwargs["timestamp"], date)

    def tearDown(self) -> None:
        shutil.rmtree(self.save_path)


class TestShortTermMemoryCleaner(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        load_dotenv()
        self.save_path = tempfile.mkdtemp()
        self.cleaner = ShortTermMemoryCleaner()
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613", temperature=0)

    async def test_end1(self):
        memory = load_memory(self.llm, self.save_path)
        add_test_messages(memory)
        memory.save_context({"input": "What is the weather today?"}, {"output": "Rainy"})
        summary = await self.cleaner.compress(memory)
        self.assertIsNotNone(summary)

    async def test_end2(self):
        memory = load_memory(self.llm, self.save_path)
        add_test_messages(memory)
        memory.save_context({"input": "hi"}, {"output": "hi"})
        summary = await self.cleaner.compress(memory)
        self.assertIsNotNone(summary)

    async def test_continue1(self):
        memory = load_memory(self.llm, self.save_path)
        add_test_messages(memory)
        summary = await self.cleaner.compress(memory)
        self.assertIsNone(summary)

    async def test_continue2(self):
        memory = load_memory(self.llm, self.save_path)
        memory.save_context({"input": "hi"}, {"output": "whats up"})
        memory.save_context({"input": "not much you"}, {"output": "not much"})
        memory.save_context({"input": "What is the weather today?"}, {"output": "Rainy"})
        summary = await self.cleaner.compress(memory)
        self.assertIsNone(summary)

    async def test_end3(self):
        memory = load_memory(self.llm, self.save_path)
        memory.save_context({"input": "hi"}, {"output": "whats up"})
        memory.save_context({"input": "not much you"}, {"output": "not much"})
        memory.save_context({"input": "hi"}, {"output": "hi"})
        self.cleaner._get_hours_after_message = lambda x: 26
        summary = await self.cleaner.compress(memory)
        self.assertIsNotNone(summary)

    async def test_compress(self):
        memory = load_memory(self.llm, self.save_path)
        add_test_messages(memory)
        memory.save_context({"input": "What is the weather today?"}, {"output": "Rainy"})
        summary = await self.cleaner.compress(memory)
        self.assertIsNotNone(summary)

    async def test_not_compress(self):
        memory = load_memory(self.llm, self.save_path)
        memory.save_context({"input": "What is the weather today?"}, {"output": "Rainy"})
        summary = await self.cleaner.compress(memory)
        self.assertEqual(len(memory.load_memory_variables({})["chat_history"]), 2)
        self.assertIsNone(summary)

    def tearDown(self) -> None:
        shutil.rmtree(self.save_path)
