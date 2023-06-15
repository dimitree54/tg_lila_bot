import shutil
import tempfile
from unittest import TestCase

from dotenv import load_dotenv
from langchain.memory.chat_memory import BaseChatMemory

from agent import Lila


class TestMemory(TestCase):
    def setUp(self) -> None:
        load_dotenv()
        self.save_path = tempfile.mkdtemp()
        self.agent = Lila(self.save_path)
        self.test_user_id = 0

    @staticmethod
    def add_test_messages(memory: BaseChatMemory):
        memory.save_context({"input": "hi"}, {"output": "whats up"})
        memory.save_context({"input": "not much you"}, {"output": "not much"})
        memory.save_context({"input": "bye"}, {"output": "see you"})

    def test_memory(self):
        memory = self.agent._load_short_term_memory(self.test_user_id)
        self.add_test_messages(memory)
        self.assertEqual(memory.load_memory_variables({})["chat_history"][0].content, "hi")

    def test_memory_savable(self):
        memory = self.agent._load_short_term_memory(self.test_user_id)
        self.add_test_messages(memory)
        memory = self.agent._load_short_term_memory(self.test_user_id)
        self.assertEqual(memory.load_memory_variables({})["chat_history"][0].content, "hi")

    def test_memory_clearable(self):
        memory = self.agent._load_short_term_memory(self.test_user_id)
        self.add_test_messages(memory)
        memory.clear()
        self.assertEqual(memory.load_memory_variables({})["chat_history"], [])
        memory = self.agent._load_short_term_memory(self.test_user_id)
        self.assertEqual(memory.load_memory_variables({})["chat_history"], [])

    def test_memory_summarization(self):
        memory = self.agent._load_short_term_memory(self.test_user_id)
        memory.max_token_limit = 20
        self.add_test_messages(memory)
        self.assertNotEqual(memory.load_memory_variables({})["chat_history"][0].content, "hi")

    def tearDown(self) -> None:
        shutil.rmtree(self.save_path)
