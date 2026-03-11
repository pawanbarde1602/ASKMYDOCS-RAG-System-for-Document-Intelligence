import streamlit as st
import pdfplumber
import tempfile

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


st.title("RAG PDF Chatbot")
st.write("Upload a PDF and ask questions from it.")


# -------- PDF UPLOADER --------

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:

    # save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    # -------- EXTRACT TEXT --------

    text = ""

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content

    # -------- TEXT SPLITTING --------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    st.write("PDF processed successfully ")

    # -------- EMBEDDINGS --------

    emb_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # -------- VECTOR DATABASE --------

    vector_db = FAISS.from_texts(
        texts=chunks,
        embedding=emb_model
    )

    retriever = vector_db.as_retriever(
        search_kwargs={"k":2}
    )

    # -------- PROMPT --------

    template = """
Answer the question based only on the following context.

Context:
{context}

Question:
{question}
"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    # -------- LLM --------

    llm = ChatGroq(
        api_key="Groq API key",
        model="llama-3.3-70b-versatile",
        temperature=0.3
    )

    # -------- RAG CHAIN --------

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # -------- QUESTION INPUT --------

    question = st.text_input("Ask a question from the PDF")

    if question:
        response = rag_chain.invoke(question)

        st.write(" Answer")
        st.write(response)