import shutil
import tempfile
from unittest import TestCase

from dotenv import load_dotenv
from langchain.llms import FakeListLLM
from langchain.memory.chat_memory import BaseChatMemory

from memory import SavableSummaryBufferMemoryWithDates


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
        llm = FakeListLLM(responses=[self.fake_summary])
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

    def tearDown(self) -> None:
        shutil.rmtree(self.save_path)
