import logging

from ddgs import DDGS

logger = logging.getLogger(__name__)


class WebSearcher:
    """
    Free, no-API-key web search via DuckDuckGo (ddgs library), used as a
    fallback when the user hasn't uploaded documents, or the uploaded
    documents don't contain anything relevant to the question.
    """

    def search(self, query: str, max_results: int = 4):
        try:
            results = DDGS().text(query, max_results=max_results)
        except Exception:
            logger.exception("Web search failed for query=%r", query)
            return []

        return [
            {
                "title": r.get("title", "Untitled"),
                "snippet": r.get("body", ""),
                "url": r.get("href", ""),
            }
            for r in results
            if r.get("body")
        ]