import asyncio
import json
from typing import Any, Tuple, List

from langchain.base_language import BaseLanguageModel
from langchain.tools import DuckDuckGoSearchResults, BaseTool
from llama_index import download_loader, GPTListIndex, Document, LLMPredictor, ServiceContext
from llama_index.response_synthesizers import TreeSummarize


class WebSearchTool(DuckDuckGoSearchResults):
    name: str = "web_search"
    description: str = \
        "Useful for when you need to search answer in the internet. " \
        "Input should be a search query (like you would google it). " \
        "If relevant, include location and date to get more accurate results. " \
        "You will get a list of urls and a short snippet of the page. "

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        return self._run(*args, **kwargs)


class AskPagesTool(BaseTool):
    llm: BaseLanguageModel
    _page_loader = download_loader("SimpleWebPageReader")(html_to_text=True)  # noqa
    name: str = "ask_urls"
    description: str = \
        "You can ask a question about a URL. " \
        "That smart tool will parse URL content and answer your question. " \
        "Provide provide urls and questions in json format. " \
        "urls is a list of urls to ask corresponding question from questions list" \
        'Example: {"urls": ["https://en.wikipedia.org/wiki/Cat", "https://en.wikipedia.org/wiki/Dog"], ' \
        '"questions": ["How many cats in the world?", "How many dogs in the world?"]}'

    def _get_page_index(self, page: Document) -> GPTListIndex:
        llm_predictor_chatgpt = LLMPredictor(self.llm)
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor_chatgpt, chunk_size=1024)
        doc_summary_index = GPTListIndex.from_documents(
            [page],
            service_context=service_context,
            response_synthesizer=TreeSummarize(service_context=service_context)
        )
        return doc_summary_index

    def _get_url_index(self, url: str) -> GPTListIndex:
        page = self._page_loader.load_data(urls=[url])[0]
        return self._get_page_index(page)

    @staticmethod
    def _parse_args(*args, **kwargs) -> List[Tuple[str, str]]:
        if len(args) == 1:
            urls_and_questions_dict = json.loads(args[0])
            urls = urls_and_questions_dict["urls"]
            questions = urls_and_questions_dict["questions"]
        else:
            urls = kwargs["urls"]
            questions = kwargs["questions"]
        return list(zip(urls, questions))

    def _run_single(self, url: str, question: str) -> str:
        page_index = self._get_url_index(url)
        llm_predictor_chatgpt = LLMPredictor(self.llm)
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor_chatgpt, chunk_size=1024)
        query_engine = page_index.as_query_engine(
            response_synthesizer=TreeSummarize(service_context=service_context), use_async=False)
        response = query_engine.query(question)
        return response.response

    async def _arun_single(self, url: str, question: str) -> str:
        page_index = self._get_url_index(url)
        llm_predictor_chatgpt = LLMPredictor(self.llm)
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor_chatgpt, chunk_size=1024)
        query_engine = page_index.as_query_engine(
            response_synthesizer=TreeSummarize(service_context=service_context), use_async=False)
        response = await query_engine.aquery(question)
        return response.response

    def _run(self, *args, **kwargs) -> Any:
        urls_with_questions = self._parse_args(*args, **kwargs)
        full_response = ""
        for url, question in urls_with_questions:
            answer = self._run_single(url, question)
            full_response += f"Question: {question} to {url}\nAnswer: {answer}\n"
        return full_response

    async def _arun(self, *args, **kwargs) -> Any:
        urls_with_questions = self._parse_args(*args, **kwargs)
        tasks = []
        for url, question in urls_with_questions:
            tasks.append(self._arun_single(url, question))
        answers = await asyncio.gather(*tasks)
        full_response = ""
        for i in range(len(urls_with_questions)):
            url, question = urls_with_questions[i]
            answer = answers[i]
            full_response += f"Question: {question} to {url}\nAnswer: {answer}\n"
        return full_response
