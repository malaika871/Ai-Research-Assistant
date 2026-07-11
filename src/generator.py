import logging

from src.services import ServiceHub
from config import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

logger = logging.getLogger(__name__)


class Generator:

    def __init__(self):
        self.client = ServiceHub.get_llm_client()

    def _format_generation_error(self, exc) -> str:
        message = str(exc).lower()
        if "api_key" in message or "token" in message or "401" in message or "403" in message:
            return "The Hugging Face inference credentials are not configured correctly. Please check the server environment."
        if "not supported" in message or ("model" in message and "supported" in message):
            return f"The configured model ({LLM_MODEL}) is currently unavailable on the Hugging Face Inference API."
        return "The language model is currently unavailable. Please try again shortly."

    # --- Intent classification (replaces keyword matching for routing) ---

    VALID_INTENTS = {"DOCUMENT", "WEB", "GENERAL"}

    def classify_intent(self, question: str) -> str:
        """
        Classifies a question into DOCUMENT, WEB, or GENERAL so the router
        can decide document-vs-web without relying on embedding similarity
        (which fails for meta-requests like "summarize this") or brittle
        keyword lists (which can't generalize to new phrasings).

        Cheap, deterministic call: temperature=0, tiny max_tokens. Falls
        back to GENERAL (the safest middle ground -- tries documents first,
        web second) if the model returns anything unparseable, so a
        classification hiccup never crashes the request.
        """
        prompt = f"""Classify the following user question into exactly one category.
Respond with ONLY one word: DOCUMENT, WEB, or GENERAL. No punctuation, no explanation.

DOCUMENT: the question asks about, summarizes, explains, or analyzes an uploaded
document (e.g. "summarize this", "what is the main contribution of this paper",
"explain section 3", "what does the author conclude", "list the key findings").

WEB: the question explicitly needs current, live, or time-sensitive external
information (e.g. "latest news", "today's weather", "current bitcoin price",
"who won today's match", "recent developments in X", "search the web").

GENERAL: a standalone knowledge question that is neither clearly about an
uploaded document nor explicitly time-sensitive (e.g. "what is machine learning").

Question: {question}
Category:"""

        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0,
            )
            raw = response.choices[0].message.content.strip().upper()
            for label in self.VALID_INTENTS:
                if label in raw:
                    return label
            logger.warning("Intent classifier returned unrecognized output=%r, defaulting to GENERAL", raw)
            return "GENERAL"
        except Exception:
            logger.exception("Intent classification failed for question=%r, defaulting to GENERAL", question)
            return "GENERAL"

    # --- Document-grounded generation ---

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
        except Exception as exc:
            logger.exception("LLM generation failed for question=%r", question)
            raise RuntimeError(self._format_generation_error(exc)) from exc

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
        except Exception as exc:
            logger.exception("LLM streaming failed for question=%r", question)
            yield f"[error] {self._format_generation_error(exc)}"
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
        except Exception as exc:
            logger.exception("LLM generation (web) failed for question=%r", question)
            raise RuntimeError(self._format_generation_error(exc)) from exc

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
        except Exception as exc:
            logger.exception("LLM streaming (web) failed for question=%r", question)
            yield f"[error] {self._format_generation_error(exc)}"
            return

        try:
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception:
            logger.exception("LLM streaming (web) interrupted for question=%r", question)
            yield "[error] The response was interrupted. Please try again."

    # --- Combined generation (document + web, used when document relevance is borderline) ---

    def _build_combined_prompt(self, question, retrieved_chunks, web_results):
        doc_context = "\n\n".join(c.text for c in retrieved_chunks)
        web_context = "\n\n".join(
            f"[{r['title']}]\n{r['snippet']}\n(Source: {r['url']})"
            for r in web_results
        )
        return f"""Answer the question using the uploaded document context first.
Only use the web search results to fill in gaps the documents don't cover.
Do not invent information beyond what's given below. If neither source has
enough information, say so honestly instead of guessing.

Document context:
{doc_context}

Web search results:
{web_context}

Question:
{question}
"""

    def generate_combined(self, question, retrieved_chunks, web_results):
        prompt = self._build_combined_prompt(question, retrieved_chunks, web_results)

        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
            )
        except Exception as exc:
            logger.exception("LLM generation (combined) failed for question=%r", question)
            raise RuntimeError(self._format_generation_error(exc)) from exc

        return {
            "answer": response.choices[0].message.content,
            "sources": [
                {"source": c.source, "page": c.page, "url": None}
                for c in retrieved_chunks
            ] + [
                {"source": r["title"], "page": None, "url": r["url"]}
                for r in web_results
            ],
            "context_type": "combined",
        }

    def generate_stream_combined(self, question, retrieved_chunks, web_results):
        prompt = self._build_combined_prompt(question, retrieved_chunks, web_results)

        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
                stream=True,
            )
        except Exception as exc:
            logger.exception("LLM streaming (combined) failed for question=%r", question)
            yield f"[error] {self._format_generation_error(exc)}"
            return

        try:
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception:
            logger.exception("LLM streaming (combined) interrupted for question=%r", question)
            yield "[error] The response was interrupted. Please try again."