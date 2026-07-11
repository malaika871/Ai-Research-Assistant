import logging

from src.services import ServiceHub
from config import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

logger = logging.getLogger(__name__)


class Generator:

    def __init__(self):
        self.client = ServiceHub.get_llm_client()

    # --- Document-grounded generation (existing path, tightened prompt) ---

    def _build_document_prompt(self, question, retrieved_chunks):
        context = "\n\n".join(c.text for c in retrieved_chunks)
        return f"""Answer the question using ONLY the context below. Do not use any
outside knowledge, and do not guess or make up information, facts, or
citations that are not present in the context.

If the context does not contain enough information to answer the question,
respond exactly with: "I don't have enough information in the uploaded
documents to answer this question."

Context:
{context}

Question:
{question}
"""

    def generate(self, question, retrieved_chunks):
        prompt = self._build_document_prompt(question, retrieved_chunks)

        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
            )
        except Exception:
            logger.exception("LLM generation failed for question=%r", question)
            raise RuntimeError(
                "The language model is currently unavailable. Please try again shortly."
            )

        return {
            "answer": response.choices[0].message.content,
            "sources": [
                {"source": c.source, "page": c.page, "url": None}
                for c in retrieved_chunks
            ],
            "context_type": "document",
        }

    def generate_stream(self, question, retrieved_chunks):
        prompt = self._build_document_prompt(question, retrieved_chunks)

        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
                stream=True,
            )
        except Exception:
            logger.exception("LLM streaming failed for question=%r", question)
            yield "[error] The language model is currently unavailable. Please try again shortly."
            return

        try:
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception:
            logger.exception("LLM streaming interrupted for question=%r", question)
            yield "[error] The response was interrupted. Please try again."

    # --- Web-grounded generation (fallback path) ---

    def _build_web_prompt(self, question, web_results):
        context = "\n\n".join(
            f"[{r['title']}]\n{r['snippet']}\n(Source: {r['url']})"
            for r in web_results
        )
        return f"""You are answering using web search results, because no relevant
uploaded documents were found for this question. Answer using ONLY the
search results below. If they don't contain enough information, say so
honestly instead of guessing.

Web search results:
{context}

Question:
{question}
"""

    def generate_from_web(self, question, web_results):
        prompt = self._build_web_prompt(question, web_results)

        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
            )
        except Exception:
            logger.exception("LLM generation (web) failed for question=%r", question)
            raise RuntimeError(
                "The language model is currently unavailable. Please try again shortly."
            )

        return {
            "answer": response.choices[0].message.content,
            "sources": [
                {"source": r["title"], "page": None, "url": r["url"]}
                for r in web_results
            ],
            "context_type": "web",
        }

    def generate_stream_from_web(self, question, web_results):
        prompt = self._build_web_prompt(question, web_results)

        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
                stream=True,
            )
        except Exception:
            logger.exception("LLM streaming (web) failed for question=%r", question)
            yield "[error] The language model is currently unavailable. Please try again shortly."
            return

        try:
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception:
            logger.exception("LLM streaming (web) interrupted for question=%r", question)
            yield "[error] The response was interrupted. Please try again."