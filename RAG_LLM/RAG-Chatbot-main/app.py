# app.py
import os
import streamlit as st
from dotenv import load_dotenv
from brain import get_index_for_pdf
from langchain_together import ChatTogether  

# Load environment variables
load_dotenv()
together_api_key = os.getenv("TOGETHER_API_KEY")

# Set Streamlit page title
st.title("📚 RAG Enhanced Chatbot with Together.ai")


@st.cache_resource   
def create_vectordb(files, filenames):
    with st.spinner("🔎 Creating vector database..."):
        vectordb = get_index_for_pdf(files, filenames)
    return vectordb


# Upload PDFs
pdf_files = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)

# Create vectordb when PDFs are uploaded
if pdf_files:
    pdf_file_names = [file.name for file in pdf_files]
    pdf_bytes = [file.read() for file in pdf_files] 
    st.session_state["vectordb"] = create_vectordb(pdf_bytes, pdf_file_names)

# Chat prompt template
prompt_template = """
You are a helpful assistant that answers user questions based on PDF content.

- Use only the given PDF extracts to answer.
- Keep answers short and precise.
- If no relevant info is found, reply: "Not applicable".

PDF Extract:
{pdf_extract}
"""

# Initialize session chat history
if "prompt" not in st.session_state:
    st.session_state["prompt"] = [{"role": "system", "content": "none"}]

prompt = st.session_state["prompt"]

# Display chat history
for message in prompt:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Get user question
question = st.chat_input("Ask your question here...")
default = " If the answer is not present in the uploaded file than just return 'there is no related info in the file.'"
if question:
    vectordb = st.session_state.get("vectordb", None)

    if not vectordb:
        with st.chat_message("assistant"):
            st.write("⚠️ Please upload at least one PDF first.")
            st.stop()

    # Perform similarity search
    search_results = vectordb.similarity_search(question+default, k=3)
    pdf_extract = "\n".join([result.page_content for result in search_results])

    # Update system prompt
    prompt[0] = {
        "role": "system",
        "content": prompt_template.format(pdf_extract=pdf_extract),
    }

    # Add user question
    prompt.append({"role": "user", "content": question+default})
    with st.chat_message("user"):
        st.write(question)

    # Response placeholder
    with st.chat_message("assistant"):
        botmsg = st.empty()

        # Call Together.ai LLM
        llm = ChatTogether(
            together_api_key=together_api_key,
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            temperature=0.3,
            max_tokens=500,
            streaming=True,
        )

        response = ""
        for chunk in llm.stream(prompt):
            if chunk.content: 
                response += chunk.content
                botmsg.write(response)

    # Save response in history
    prompt.append({"role": "assistant", "content": response})
    st.session_state["prompt"] = prompt
