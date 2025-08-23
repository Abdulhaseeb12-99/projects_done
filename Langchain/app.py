import customtkinter as ctk
import pandas as pd
from tkinter import filedialog, messagebox
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPDFLoader,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
from langchain_together import Together
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
import subprocess
import platform

# 🔐 Replace with your Together API key
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOGETHER_API_KEY"] = "b11c52b3b493ba408f0a86cf7021296cda922913890788ab38670cb779e8690b"
chat_history = []

# Configure theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class ChatBotApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("LangChain Document Chatbot")
        self.geometry("800x600")
        self.file_path = None
        self.db = None

        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=150)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)

        self.upload_button = ctk.CTkButton(self.sidebar, text="Upload Document", command=self.load_file)
        self.upload_button.pack(pady=(20, 10))

        self.view_button = ctk.CTkButton(self.sidebar, text="View Document", command=self.view_file)
        self.view_button.pack(pady=(10, 0))
        self.view_button.configure(state="disabled")

        self.knowledge_switch = ctk.CTkSwitch(
            self.sidebar,
            text="Enable External Knowledge"
        )
        self.knowledge_switch.pack(pady=(10, 0))

        self.status_label = ctk.CTkLabel(self.sidebar, text="No file loaded", wraplength=140, justify="left")
        self.status_label.pack(pady=(10, 0))

        # Main display
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.chat_display = ctk.CTkTextbox(self.main_frame, wrap="word")
        self.chat_display.pack(fill="both", expand=True, padx=10, pady=(10, 5))
        self.chat_display.configure(state="disabled")

        self.input_frame = ctk.CTkFrame(self.main_frame)
        self.input_frame.pack(fill="x", padx=10, pady=(0, 10))

        self.input_box = ctk.CTkEntry(self.input_frame, placeholder_text="Ask a question...")
        self.input_box.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.send_button = ctk.CTkButton(self.input_frame, text="Send", command=self.ask_question)
        self.send_button.pack(side="right")

    def load_file(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            messagebox.showerror("Error", "File not found!")
            return

        try:
            if file_path.endswith(".pdf"):
                try:
                    loader = PyPDFLoader(file_path)
                    raw_docs = loader.load()
                    if not raw_docs or not isinstance(raw_docs, list) or not all(hasattr(doc, "page_content") for doc in raw_docs):
                        raise ValueError("Empty or invalid PDF content")
                except Exception:
                    try:
                        loader = UnstructuredPDFLoader(file_path, mode="elements")
                        raw_docs = loader.load()
                        if not raw_docs:
                            messagebox.showerror("Error", "PDF is scanned or unreadable, and OCR failed.")
                            return
                        messagebox.showinfo("Note", "Used OCR to read scanned PDF.")
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to process PDF with OCR: {e}")
                        return
            elif file_path.endswith(".txt"):
                loader = TextLoader(file_path)
                raw_docs = loader.load()
            elif file_path.endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(file_path)
                raw_docs = loader.load()
            elif file_path.endswith(".csv"):
                df = pd.read_csv(file_path)
                text_data = df.to_string(index=False)
                raw_docs = [Document(page_content=text_data)]
            elif file_path.endswith(".xlsx"):
                df = pd.read_excel(file_path)
                text_data = df.to_string(index=False)
                raw_docs = [Document(page_content=text_data)]
            else:
                messagebox.showerror("Unsupported File", "Please select a PDF, TXT, DOCX, CSV, or XLSX file.")
                return

            self.file_path = file_path
            self.view_button.configure(state="normal")

            if not raw_docs:
                messagebox.showerror("Error", "Failed to read the document or it's empty.")
                return

            # Split
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            documents = splitter.split_documents(raw_docs)

            # Embed & store
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            self.db = FAISS.from_documents(documents, embeddings)

            self.chat_display.configure(state="normal")
            self.chat_display.insert("end", f"\nLoaded file: {os.path.basename(file_path)}\n")
            self.chat_display.configure(state="disabled")

            messagebox.showinfo("Success", "File successfully processed!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")

    def view_file(self):
        if self.file_path:
            try:
                if platform.system() == "Windows":
                    os.startfile(self.file_path)
                elif platform.system() == "Darwin":
                    subprocess.call(["open", self.file_path])
                else:
                    subprocess.call(["xdg-open", self.file_path])
            except Exception as e:
                messagebox.showerror("Error", f"Could not open file: {e}")

    def ask_question(self):
        input_question = self.input_box.get().strip()
        question = self.input_box.get().strip() + " Note that if the answer is not present in the document then just return 'I don't know.'or 'Not in the text.'."
        if not question:
            messagebox.showwarning("Warning", "Enter a question.")
            return

        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", f"\nYou: {input_question}\n")
        self.chat_display.configure(state="disabled")
        self.input_box.delete(0, "end")

        try:
            llm = Together(
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                temperature=0.3,
                max_tokens=512
            )

            answer = "I don't know."
            doc_answer = ""
            relevant_docs = []

            # Step 1: If a document is loaded, search for relevant docs
            if self.db:
                relevant_docs = self.db.similarity_search(question, k=3)

            # Step 2: If docs exist, try to answer from the document
            if relevant_docs:
                chain = load_qa_chain(llm, chain_type="stuff")
                doc_result = chain.invoke({
                    "input_documents": relevant_docs,
                    "question": question
                })
                doc_answer = doc_result.get("output_text", "").strip()
                
                # Check if the document provided a meaningful answer
                if (doc_answer and 
                    "I don't know." not in doc_answer.lower() and
                    "Not in the text." not in doc_answer.lower() and
                    "not mentioned" not in doc_answer.lower() and
                    len(doc_answer) > 10):  # Minimum length to avoid short non-answers
                    answer = f"Based on the document: {doc_answer}"
            
            # Step 3: If no document answer was found, check external knowledge switch
            if (answer == "I don't know." or "Not in the text.") and self.knowledge_switch.get() == 1:
                try:
                    ext_answer = llm.invoke(question)
                    answer = f"Based on general knowledge: {ext_answer.strip()}" if ext_answer else "I don't know."
                except Exception as e:
                    answer = "I encountered an error accessing external knowledge."

            # Step 4: If still no answer and no external knowledge, return "I don't know"
            chat_history.append(f"You: {question}")
            chat_history.append(f"Bot: {answer}")

            self.chat_display.configure(state="normal")
            self.chat_display.insert("end", f"Bot: {answer}\n")
            self.chat_display.see("end")
            self.chat_display.configure(state="disabled")

        except Exception as e:
            messagebox.showerror("Error", f"Something went wrong: {e}")



if __name__ == "__main__":
    app = ChatBotApp()
    app.mainloop()
