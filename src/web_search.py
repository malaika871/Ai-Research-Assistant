import logging

try:
    from duckduckgo_search import DDGS
except ImportError:  # pragma: no cover - fallback for older environments
    from ddgs import DDGS  # type: ignore

logger = logging.getLogger(__name__)


class WebSearcher:
    """
    Free, no-API-key web search via DuckDuckGo, used as a fallback when the
    user hasn't uploaded documents, or the uploaded documents don't contain
    anything relevant to the question.
    """

    def search(self, query: str, max_results: int = 4):
        if DDGS is None:
            logger.warning("DuckDuckGo search client is unavailable; web search disabled")
            return []

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