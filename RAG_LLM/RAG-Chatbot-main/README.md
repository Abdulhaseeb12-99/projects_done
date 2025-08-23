This project is a Retrieval-Augmented Generation (RAG) chatbot built using LangChain, FAISS, Together.ai LLMs, and Streamlit. It allows users to upload one or more PDF documents and then ask natural language questions about their content.

PDF Upload – Users upload PDFs through the Streamlit interface.

Vector Indexing – The PDFs are parsed, split into chunks, and stored in a FAISS vector database using Together.ai embeddings.

Question Answering (RAG) – When the user asks a question, the system retrieves the most relevant chunks from the vector DB and feeds them into the LLM (e.g., Mixtral-8x7B-Instruct).

Context-Aware Response – The LLM generates precise, short answers based only on the extracted PDF content.