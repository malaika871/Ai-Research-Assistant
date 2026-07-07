from src.services import ServiceHub
from config import LLM_MODEL


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

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.2,
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

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.2,
            stream=True,
        )

        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield content