import logging

from src.services import ServiceHub
from config import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

logger = logging.getLogger(__name__)


class Generator:

    def __init__(self):
        self.client = ServiceHub.get_llm_client()

    def generate(self, question, retrieved_chunks):

        context = "\n\n".join(
            c.text for c in retrieved_chunks
        )

        prompt = f"""
Answer using context only.

Context:
{context}

Question:
{question}
"""

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
                {
                    "source": c.source,
                    "page": c.page
                }
                for c in retrieved_chunks
            ]
        }

    def generate_stream(self, question, retrieved_chunks):
        context = "\n\n".join(
            c.text for c in retrieved_chunks
        )

        prompt = f"Answer using context only.\n\nContext:\n{context}\n\nQuestion:\n{question}\n"

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
import logging

from src.services import ServiceHub
from config import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

logger = logging.getLogger(__name__)


class Generator:

    def __init__(self):
        self.client = ServiceHub.get_llm_client()

    def generate(self, question, retrieved_chunks):

        context = "\n\n".join(
            c.text for c in retrieved_chunks
        )

        prompt = f"""
Answer using context only.

Context:
{context}

Question:
{question}
"""

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
                {
                    "source": c.source,
                    "page": c.page
                }
                for c in retrieved_chunks
            ]
        }

    def generate_stream(self, question, retrieved_chunks):
        context = "\n\n".join(
            c.text for c in retrieved_chunks
        )

        prompt = f"Answer using context only.\n\nContext:\n{context}\n\nQuestion:\n{question}\n"

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
        try:
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception:
            logger.exception("LLM streaming interrupted for question=%r", question)
            yield "[error] The response was interrupted. Please try again."