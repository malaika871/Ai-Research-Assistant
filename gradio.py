import os
from dotenv import load_dotenv

# Load environment variables from src/.env
base_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(base_dir, "src", ".env")
load_dotenv(dotenv_path)

import gradio as gr
from src.rag_engine import RAGEngine

# Initialize engine
engine = RAGEngine()


def upload_documents(files):
    """Index uploaded documents."""

    if not files:
        return "Please upload at least one document."

    file_paths = [file.name for file in files]

    result = engine.index_documents(file_paths)

    return (
        f"✅ Indexed {result['indexed_files']} document(s)\n"
        f"📄 Created {result['total_chunks']} chunks."
    )


def ask_question(question):
    """Answer user question."""

    if not question.strip():
        return "Please enter a question."

    response = engine.ask(question)

    # If generator returns dict
    if isinstance(response, dict):
        return response["answer"]

    # Otherwise return string
    return response


def list_documents():
    docs = engine.list_documents()

    if not docs:
        return "No indexed documents."

    return "\n".join(docs)


def delete_document(document_name):

    if not document_name.strip():
        return "Enter document name."

    engine.delete_document(document_name)

    return f"Deleted {document_name}"


with gr.Blocks(title="AI Research Assistant") as demo:

    gr.Markdown(
        """
        # 📚 AI Research Assistant

        Upload multiple research papers and ask questions using RAG.
        """
    )

    with gr.Tab("📂 Upload"):

        uploader = gr.File(
            file_count="multiple",
            file_types=[".pdf", ".docx", ".txt"],
            label="Upload Documents"
        )

        upload_btn = gr.Button("Index Documents")

        upload_output = gr.Textbox(
            label="Status",
            lines=4
        )

        upload_btn.click(
            fn=upload_documents,
            inputs=uploader,
            outputs=upload_output
        )

    with gr.Tab("💬 Chat"):

        question = gr.Textbox(
            label="Ask a Question",
            placeholder="What is self-attention?"
        )

        ask_btn = gr.Button("Ask")

        answer = gr.Textbox(
            label="Answer",
            lines=12
        )

        ask_btn.click(
            fn=ask_question,
            inputs=question,
            outputs=answer
        )

    with gr.Tab("📑 Documents"):

        refresh_btn = gr.Button("Refresh")

        docs_box = gr.Textbox(
            label="Indexed Documents",
            lines=10
        )

        refresh_btn.click(
            fn=list_documents,
            outputs=docs_box
        )

    with gr.Tab("🗑 Delete"):

        delete_input = gr.Textbox(
            label="Document Name",
            placeholder="attention.pdf"
        )

        delete_btn = gr.Button("Delete")

        delete_output = gr.Textbox(
            label="Status"
        )

        delete_btn.click(
            fn=delete_document,
            inputs=delete_input,
            outputs=delete_output
        )


demo.launch()