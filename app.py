import streamlit as st
import tempfile
import os
from rag_engine import AskMyDocsEngine

# Page Config
st.set_page_config(page_title="AskMyDocs – LLM-Powered RAG Document QA System", layout="wide", initial_sidebar_state="expanded")

# CSS for a premium look
st.markdown("""
    <style>
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .stButton>button {
        background: linear-gradient(135deg, #2ea44f 0%, #238636 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        color: white;
    }
    .chat-bubble {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    .user-bubble {
        background-color: #21262d;
        border-left: 5px solid #58a6ff;
    }
    .ai-bubble {
        background-color: #161b22;
        border-left: 5px solid #2ea44f;
    }
    </style>
    """, unsafe_allow_html=True)

# Application Logo and Description
st.title("📄 AskMyDocs")
st.markdown("Retrieval-Augmented Generation for Q&A over private document collections.")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    
    st.divider()
    st.subheader("🛠 Pipeline Settings")
    
    # Chunking Strategy as requested (500/1000/1500 tokens)
    chunking_strategy = st.radio("Chunking Strategy", ["Fixed (Recursive)", "Token Splitting"])
    chunk_size = st.select_slider("Chunk Size", options=[500, 1000, 1500], value=1000)
    
    # Temperature as requested (0.3 vs 0.7)
    temp_choice = st.radio("Model Halucination Control (Temperature)", [0.3, 0.7], index=0, help="Lower temperature reduces hallucinations.")
    
    st.divider()
    uploaded_file = st.file_uploader("Upload PDF File", type=["pdf"])

# Main Application Logic
if uploaded_file and api_key:
    if "engine" not in st.session_state:
        st.session_state.engine = AskMyDocsEngine(api_key)
        
        # Save temp file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        with st.spinner("Processing Document... (Ingestion -> Chunking -> Embeddings)"):
            strategy = "fixed" if chunking_strategy == "Fixed (Recursive)" else "token"
            chunks = st.session_state.engine.process_pdf(tmp_path, chunk_strategy=strategy, chunk_size=chunk_size)
            vectorstore = st.session_state.engine.create_vector_store(chunks)
            st.session_state.qa_chain = st.session_state.engine.get_qa_chain(vectorstore, temperature=temp_choice)
            st.success("Document analyzed and ready for Q&A!")
            os.unlink(tmp_path)

    # Chat Interface
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.divider()
    
    # Container for Chat History
    chat_container = st.container()
    
    # Question Input
    user_query = st.chat_input("Ask a question about your document...")

    if user_query:
        # User Bubble
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Generator AI Response
        with st.spinner("Generating Answer..."):
            response = st.session_state.qa_chain.invoke({"question": user_query, "chat_history": []})
            st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})

    # Render Chat History
    for chat in st.session_state.chat_history:
        role_label = "👤 You" if chat["role"] == "user" else "🤖 AskMyDocs"
        bubble_class = "user-bubble" if chat["role"] == "user" else "ai-bubble"
        st.markdown(f"""
            <div class="chat-bubble {bubble_class}">
                <strong>{role_label}:</strong><br/>
                {chat["content"]}
            </div>
        """, unsafe_allow_html=True)

elif not uploaded_file:
    st.info("Please upload a PDF document and enter your API key to get started.")
