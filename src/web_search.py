import logging
import os

from tavily import TavilyClient

logger = logging.getLogger(__name__)


class WebSearcher:
    """
    Web search via Tavily -- an API built specifically for LLM/RAG use
    cases, more reliable than scraping-based alternatives (which can get
    served degraded/generic results from datacenter IPs like HF Spaces).

    Requires TAVILY_API_KEY in the environment. Get a free-tier key at
    https://tavily.com
    """

    def __init__(self):
        api_key = os.getenv("TAVILY_API_KEY")
        self._client = TavilyClient(api_key=api_key) if api_key else None
        if not api_key:
            logger.warning("TAVILY_API_KEY not set -- web search will return no results.")

    def search(self, query: str, max_results: int = 4):
        if not self._client:
            logger.error("Web search called but TAVILY_API_KEY is not configured.")
            return []

        try:
            response = self._client.search(query, max_results=max_results)
        except Exception as exc:
            logger.error(
                "Tavily web search failed for query=%r: %s: %s",
                query, type(exc).__name__, exc,
            )
            return []

        results = response.get("results", []) if isinstance(response, dict) else []

        return [
            {
                "title": r.get("title", "Untitled"),
                "snippet": r.get("content", ""),
                "url": r.get("url", ""),
            }
            for r in results
            if r.get("content")
        ]