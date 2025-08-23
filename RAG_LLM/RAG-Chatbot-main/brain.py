import os
import re
from io import BytesIO
from typing import Tuple, List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_together import TogetherEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# Load environment variables from .env
load_dotenv()

# Get Together API key from environment
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY not found in environment variables. Please set it in your .env file.")


def parse_pdf(file: BytesIO, filename: str) -> Tuple[List[str], str]:
    """Extracts and cleans text from PDF pages."""
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        if not text:
            continue
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)  # join broken words
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())  # fix newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)  # preserve paragraphs
        output.append(text)
    return output, filename


def text_to_docs(text: List[str], filename: str) -> List[Document]:
    """Splits text into smaller chunks and converts to LangChain Documents."""
    if isinstance(text, str):
        text = [text]
    page_docs = [Document(page_content=page, metadata={"page": i + 1}) for i, page in enumerate(text)]

    doc_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )

    for doc in page_docs:
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            new_doc = Document(
                page_content=chunk,
                metadata={
                    "page": doc.metadata["page"],
                    "chunk": i,
                    "source": f"{doc.metadata['page']}-{i}",
                    "filename": filename,
                },
            )
            doc_chunks.append(new_doc)
    return doc_chunks


def docs_to_index(docs: List[Document]):
    """Converts documents into a FAISS vector index using Together embeddings."""
    embeddings = TogetherEmbeddings(
       model="togethercomputer/m2-bert-80M-32k-retrieval",
        api_key=TOGETHER_API_KEY
    )
    index = FAISS.from_documents(docs, embeddings)
    return index


def get_index_for_pdf(pdf_files: List[bytes], pdf_names: List[str]):
    """Parses multiple PDFs and builds a FAISS index."""
    documents = []
    for pdf_file, pdf_name in zip(pdf_files, pdf_names):
        text, filename = parse_pdf(BytesIO(pdf_file), pdf_name)
        documents.extend(text_to_docs(text, filename))
    index = docs_to_index(documents)
    return index
