import shutil
import tempfile
from typing import List
from unittest import TestCase

import tiktoken
from dotenv import load_dotenv
from langchain.llms import FakeListLLM
from langchain.memory.chat_memory import BaseChatMemory

from memory import SavableSummaryBufferMemoryWithDates


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

    @staticmethod
    def add_test_messages(memory: BaseChatMemory):
        memory.save_context({"input": "hi"}, {"output": "whats up"})
        memory.save_context({"input": "not much you"}, {"output": "not much"})
        memory.save_context({"input": "bye"}, {"output": "see you"})

    def load_memory(self, token_limit: int = 6000) -> SavableSummaryBufferMemoryWithDates:
        llm = FakeListLLMTiktoken(responses=[self.fake_summary])
        return SavableSummaryBufferMemoryWithDates.load(
            llm=llm, max_token_limit=token_limit,
            memory_key="chat_history", return_messages=True,
            save_path=self.save_path
        )

    def test_memory(self):
        memory = self.load_memory()
        self.add_test_messages(memory)
        self.assertEqual(memory.load_memory_variables({})["chat_history"][0].content, "hi")

    def test_memory_savable(self):
        memory = self.load_memory()
        self.add_test_messages(memory)
        memory = self.load_memory()
        self.assertEqual(memory.load_memory_variables({})["chat_history"][0].content, "hi")

    def test_memory_clearable(self):
        memory = self.load_memory()
        self.add_test_messages(memory)
        memory.clear()
        self.assertEqual(memory.load_memory_variables({})["chat_history"], [])
        self.assertEqual(memory.moving_summary_buffer, "")
        self.assertEqual(memory.chat_memory.messages, [])
        memory = self.load_memory()
        self.assertEqual(memory.load_memory_variables({})["chat_history"], [])
        self.assertEqual(memory.moving_summary_buffer, "")
        self.assertEqual(memory.chat_memory.messages, [])

    def test_memory_summarization(self):
        memory = self.load_memory(20)
        self.add_test_messages(memory)
        self.assertEqual(memory.load_memory_variables({})["chat_history"][0].content, self.fake_summary)

    def test_memory_summarization_savable(self):
        memory = self.load_memory(20)
        self.add_test_messages(memory)
        memory = self.load_memory(20)
        self.assertEqual(memory.load_memory_variables({})["chat_history"][0].content, self.fake_summary)

    def test_date(self):
        memory = self.load_memory()
        self.add_test_messages(memory)
        self.assertTrue("timestamp" in memory.load_memory_variables({})["chat_history"][0].additional_kwargs)

    def test_date_savable(self):
        memory = self.load_memory()
        self.add_test_messages(memory)
        date = memory.load_memory_variables({})["chat_history"][0].additional_kwargs["timestamp"]
        memory = self.load_memory()
        self.assertEqual(memory.load_memory_variables({})["chat_history"][0].additional_kwargs["timestamp"], date)

    def tearDown(self) -> None:
        shutil.rmtree(self.save_path)
