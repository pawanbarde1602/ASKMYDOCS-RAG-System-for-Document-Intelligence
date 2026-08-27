import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os

# ---------------- CONFIG ----------------
st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="📄")

# Put your real keys in .streamlit/secrets.toml (see instructions below)
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
INDEX_NAME = "my-index"

# ---------------- CACHED RESOURCES (load once, not on every rerun) ----------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(INDEX_NAME)

@st.cache_resource
def load_llm():
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(
        "gemini-2.5-flash",
        generation_config={"temperature": 0.1}
    )

model = load_embedding_model()
index = load_pinecone_index()
model_llm = load_llm()

# ---------------- RAG FUNCTION ----------------
SCORE_THRESHOLD = 0.3  # matches below this score are treated as irrelevant

def ask(query_text, top_k=5):
    query_emb = model.encode(query_text).tolist()
    results = index.query(vector=query_emb, top_k=top_k, include_metadata=True)
    matches = results["matches"]

    relevant_matches = [m for m in matches if m["score"] >= SCORE_THRESHOLD]

    if not relevant_matches:
        return "I couldn't find relevant information in the document for this question.", matches

    context = "\n".join([m["metadata"]["text"] for m in relevant_matches])

    prompt = f"""You are a helpful assistant that answers ONLY using the provided context.

Rules:
- If the answer is not in the context, say "I don't have information about this in the document."
- Do NOT use outside knowledge or make assumptions.
- Do NOT guess or infer beyond what's explicitly stated.

Context:
{context}

Question: {query_text}

Answer:"""

    response = model_llm.generate_content(prompt)
    return response.text, relevant_matches

# ---------------- PDF UPLOAD + PROCESSING ----------------
def process_pdf(uploaded_file):
    # save uploaded file to a temp path so PyPDFLoader can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(pages)

    texts = [c.page_content for c in chunks]
    embeddings = model.encode(texts).tolist()

    vectors = []
    for i, (text, emb) in enumerate(zip(texts, embeddings)):
        vectors.append({
            "id": f"{uploaded_file.name}-chunk{i}",
            "values": emb,
            "metadata": {"text": text, "source": uploaded_file.name}
        })

    # upsert in batches of 100
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i + batch_size])

    os.remove(tmp_path)
    return len(vectors)

# ---------------- UI ----------------
st.title("📄 PDF Q&A Chatbot")
st.caption("Upload a PDF, then ask questions about it")

with st.expander("📤 Upload a new PDF", expanded=False):
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        if st.button("Process & Upload to Pinecone"):
            with st.spinner("Reading, chunking, and embedding PDF... this may take a minute"):
                num_chunks = process_pdf(uploaded_file)
            st.success(f"Uploaded {num_chunks} chunks from '{uploaded_file.name}' to Pinecone!")

# keep chat history across reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

# render past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# chat input
user_input = st.chat_input("Ask a question...")

if user_input:
    # show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, matches = ask(user_input)
            st.markdown(answer)

            with st.expander("Sources used"):
                for m in matches:
                    st.write(f"**Score:** {m['score']:.3f}")
                    st.write(m["metadata"]["text"])
                    st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})

# sidebar
with st.sidebar:
    st.header("Settings")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("⚠️ Document Management")
    st.caption("Clear old PDF data before uploading a new document, to avoid mixed answers.")
    if st.button("🗑️ Clear all documents from Pinecone"):
        index.delete(delete_all=True)
        st.session_state.messages = []
        st.success("All documents cleared from Pinecone!")
        st.rerun()