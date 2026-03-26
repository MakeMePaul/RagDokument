"""
RAG (Retrieval-Augmented Generation) Pipeline

A local script that reads a PDF document, splits it into chunks, stores them
in a ChromaDB vector database, and provides an interactive terminal chat
interface for asking questions about the document.
"""

import os
import sys

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

CHROMA_PERSIST_DIR = "./chroma_db"


def load_document(file_path: str) -> list[Document]:
    """Load a PDF file and return a list of Document objects (one per page)."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"The PDF file was not found at: {file_path}"
        )

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    if not documents:
        raise ValueError(f"The PDF at '{file_path}' produced no readable content.")

    print(f"Loaded {len(documents)} page(s) from '{file_path}'.")
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunk(s).")
    return chunks


def setup_vectorstore(chunks: list[Document]) -> Chroma:
    """Create (or overwrite) a ChromaDB vector store from document chunks."""
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print(f"Vector store created with {len(chunks)} vectors in '{CHROMA_PERSIST_DIR}'.")
    return vectorstore


def build_rag_chain(vectorstore: Chroma):
    """Build a LangChain retrieval chain backed by the vector store."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    system_prompt = (
        "You are a helpful assistant that answers questions based on the "
        "provided context extracted from a PDF document. If the answer cannot "
        "be found in the context, say so clearly instead of making something up.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain


def main() -> None:
    """Entry point: load PDF, build pipeline, and start interactive chat."""
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Error: OPENAI_API_KEY is not set. "
            "Create a .env file (see .env.example) and add your key."
        )
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Usage: python rag_pipeline.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    try:
        documents = load_document(pdf_path)
        chunks = split_documents(documents)
        vectorstore = setup_vectorstore(chunks)
        rag_chain = build_rag_chain(vectorstore)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error during setup: {exc}")
        sys.exit(1)

    print("\nRAG pipeline is ready. Ask questions about your document.")
    print("Type 'exit' to quit.\n")

    try:
        while True:
            question = input("You: ").strip()

            if not question:
                continue
            if question.lower() == "exit":
                print("Goodbye!")
                break

            try:
                result = rag_chain.invoke({"input": question})
                print(f"\nAssistant: {result['answer']}\n")
            except Exception as exc:
                print(f"\nAn error occurred while processing your question: {exc}\n")

    except KeyboardInterrupt:
        print("\nSession interrupted. Goodbye!")


if __name__ == "__main__":
    main()
