import json
from typing import Any, Tuple

from langchain.base_language import BaseLanguageModel
from langchain.tools import DuckDuckGoSearchResults, BaseTool
from llama_index import download_loader, GPTListIndex, Document, LLMPredictor, ServiceContext, ResponseSynthesizer
from llama_index.indices.response import ResponseMode


class WebSearchTool(DuckDuckGoSearchResults):
    name: str = "web_search"
    description: str = \
        "Useful for when you need to search answer in the internet. " \
        "Input should be a search query (like you would google it). " \
        "If relevant, include location and date to get more accurate results. " \
        "You will get a list of urls and a short snippet of the page. "

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        return self._run(*args, **kwargs)


class AskPageTool(BaseTool):
    llm: BaseLanguageModel
    _page_loader = download_loader("SimpleWebPageReader")(html_to_text=True)  # noqa
    name: str = "ask_url"
    description: str = \
        "You can ask a question about a URL. " \
        "That smart tool will parse URL content and answer your question. " \
        "Provide url and question in json format. " \
        "Example: {'url': 'https://en.wikipedia.org/wiki/Cat', 'question': 'How many cats in the world?'}"

    def _get_page_index(self, page: Document) -> GPTListIndex:
        llm_predictor_chatgpt = LLMPredictor(self.llm)
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor_chatgpt, chunk_size=1024)
        response_synthesizer = ResponseSynthesizer.from_args(response_mode=ResponseMode.TREE_SUMMARIZE, use_async=True)
        doc_summary_index = GPTListIndex.from_documents(
            [page],
            service_context=service_context,
            response_synthesizer=response_synthesizer
        )
        return doc_summary_index

    def _get_url_index(self, url: str) -> GPTListIndex:
        page = self._page_loader.load_data(urls=[url])[0]
        return self._get_page_index(page)

    @staticmethod
    def _parse_args(*args, **kwargs) -> Tuple[str, str]:
        if len(args) == 1:
            url_and_request_dict = json.loads(args[0])
            url = url_and_request_dict["url"]
            question = url_and_request_dict["question"]
        else:
            url = kwargs["url"]
            question = kwargs["question"]
        return url, question

    def _run(self, *args, **kwargs) -> Any:
        url, question = self._parse_args(*args, **kwargs)
        page_index = self._get_url_index(url)
        query_engine = page_index.as_query_engine(response_mode=ResponseMode.TREE_SUMMARIZE, use_async=False)
        response = query_engine.query(question)
        return response.response

    async def _arun(self, *args, **kwargs) -> Any:
        url, question = self._parse_args(*args, **kwargs)
        page_index = self._get_url_index(url)
        query_engine = page_index.as_query_engine(response_mode=ResponseMode.TREE_SUMMARIZE, use_async=False)
        response = await query_engine.aquery(question)
        return response.response
