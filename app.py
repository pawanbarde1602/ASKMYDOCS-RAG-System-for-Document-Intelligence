# AVG's Web Shield intercepts HTTPS and re-signs certs with its own root CA, which
# certifi doesn't trust -> CERTIFICATE_VERIFY_FAILED on every outbound call.
# truststore makes Python use the Windows cert store, which does trust it.
# Must run before anything creates an SSL context.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from google import genai
from google.genai import types
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os
import re

# ---------------- CONFIG ----------------
st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="📄")

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
INDEX_NAME = "my-index"
LLM_MODEL = "gemini-2.5-flash"

# Retrieval tuning
TOP_K = 20            # retrieve wide, let the LLM filter
MIN_SCORE = 0.15      # absolute floor: below this a cosine match is noise
RELATIVE_CUT = 0.55   # keep matches scoring >= 55% of the best match
MAX_CONTEXT_CHUNKS = 12

# ---------------- CACHED RESOURCES ----------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(INDEX_NAME)

@st.cache_resource
def load_llm_client():
    return genai.Client(api_key=GEMINI_API_KEY)

model = load_embedding_model()
index = load_pinecone_index()
client = load_llm_client()


# ---------------- LLM HELPER ----------------
def generate(prompt, max_tokens=2048):
    """Call Gemini with thinking disabled, and never blow up on an empty response."""
    resp = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=max_tokens,
            # gemini-2.5-* models think by default; thinking tokens eat the output
            # budget and can leave zero text parts behind. Off for extractive QA.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )

    if not resp.candidates:
        return None, "blocked_or_empty"

    cand = resp.candidates[0]
    parts = getattr(cand.content, "parts", None) or []
    text = "".join(p.text for p in parts if getattr(p, "text", None))

    if not text.strip():
        return None, str(getattr(cand, "finish_reason", "unknown"))

    return text.strip(), None


# ---------------- RETRIEVAL ----------------
def embed(texts):
    """Normalized embeddings - required for dotproduct indexes, harmless for cosine."""
    return model.encode(texts, normalize_embeddings=True).tolist()


def condense_question(query_text, history):
    """Rewrite a follow-up into a standalone question so retrieval has something to match on.

    "What about his education?" embeds to nothing useful on its own.
    """
    if not history:
        return query_text

    recent = history[-6:]
    convo = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)

    prompt = f"""Rewrite the follow-up question as a standalone question that makes sense without the conversation.
Keep it short. Resolve pronouns and references using the conversation. Output ONLY the rewritten question.

Conversation:
{convo}

Follow-up question: {query_text}

Standalone question:"""

    rewritten, err = generate(prompt, max_tokens=256)
    if err or not rewritten:
        return query_text
    return rewritten.split("\n")[0].strip()


def retrieve(search_query, namespace):
    results = index.query(
        vector=embed([search_query])[0],
        top_k=TOP_K,
        include_metadata=True,
        namespace=namespace,
    )
    matches = results.get("matches", [])
    if not matches:
        return []

    best = matches[0]["score"]
    kept = [
        m for m in matches
        if m["score"] >= MIN_SCORE and m["score"] >= best * RELATIVE_CUT
    ]
    return kept[:MAX_CONTEXT_CHUNKS]


def build_context(matches):
    blocks = []
    for i, m in enumerate(matches, 1):
        meta = m.get("metadata", {}) or {}
        page = meta.get("page", "?")
        text = meta.get("text", "")
        blocks.append(f"[{i}] (page {page})\n{text}")
    return "\n\n---\n\n".join(blocks)


NOT_FOUND_TOKEN = "NOT_IN_DOCUMENT"


def ask(query_text, namespace, history):
    search_query = condense_question(query_text, history)
    matches = retrieve(search_query, namespace)

    if not matches:
        return ("I couldn't find anything related to that in this document.",
                [], search_query)

    context = build_context(matches)

    prompt = f"""You are a document question-answering assistant. Answer using ONLY the numbered passages below, which were retrieved from the user's PDF.

Rules:
- Use only the passages. Never add outside knowledge or invent details.
- You MAY combine, compare and summarize information across passages to build a complete answer.
- Cite the passages you used inline, like [1] or [2][5].
- If the passages genuinely do not contain the answer, reply with exactly: {NOT_FOUND_TOKEN}
- Otherwise answer directly and concisely. Do not mention "the passages" or "the context" in your wording.

Passages:
{context}

Question: {search_query}

Answer:"""

    answer, err = generate(prompt)

    if err:
        return (f"⚠️ The model returned no text (finish_reason: `{err}`). "
                f"Try rephrasing, or ask a narrower question.", matches, search_query)

    if NOT_FOUND_TOKEN in answer:
        return ("I don't have information about this in the document.", matches, search_query)

    return answer, matches, search_query


# ---------------- PDF UPLOAD + PROCESSING ----------------
def make_namespace(filename):
    """Pinecone namespaces allow alphanumerics, '-' and '_'."""
    ns = re.sub(r"[^A-Za-z0-9_-]", "-", os.path.splitext(filename)[0])
    return ns.strip("-")[:48] or "default"


def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        extracted = sum(len(p.page_content.strip()) for p in pages)
        if extracted < 50 * max(len(pages), 1):
            # A scanned/image PDF yields almost no text. Uploading it produces an
            # index full of blanks, which is what makes answers look hallucinated.
            return None, 0, (
                f"Only {extracted} characters of text were extracted from {len(pages)} page(s). "
                "This PDF is probably scanned images and needs OCR before it can be indexed."
            )

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(pages)
        chunks = [c for c in chunks if len(c.page_content.strip()) > 30]

        if not chunks:
            return None, 0, "No usable text chunks were produced from this PDF."

        texts = [c.page_content for c in chunks]
        embeddings = embed(texts)
        namespace = make_namespace(uploaded_file.name)

        vectors = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            vectors.append({
                "id": f"{namespace}-chunk{i}",
                "values": emb,
                "metadata": {
                    "text": chunk.page_content,
                    "source": uploaded_file.name,
                    "page": chunk.metadata.get("page", 0) + 1,
                },
            })

        # replace any previous version of this document
        try:
            index.delete(delete_all=True, namespace=namespace)
        except Exception:
            pass  # namespace may not exist yet

        for i in range(0, len(vectors), 100):
            index.upsert(vectors=vectors[i:i + 100], namespace=namespace)

        return namespace, len(vectors), None
    finally:
        os.remove(tmp_path)


def list_documents():
    try:
        stats = index.describe_index_stats()
        return sorted(
            ns for ns, info in (stats.get("namespaces") or {}).items()
            if info.get("vector_count", 0) > 0
        )
    except Exception as e:
        st.sidebar.error(f"Could not read index: {e}")
        return []


# ---------------- UI ----------------
st.title("📄 PDF Q&A Chatbot")
st.caption("Upload a PDF, then ask questions about it")

with st.expander("📤 Upload a new PDF", expanded=False):
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        if st.button("Process & Upload to Pinecone"):
            with st.spinner("Reading, chunking, and embedding PDF..."):
                namespace, num_chunks, error = process_pdf(uploaded_file)
            if error:
                st.error(error)
            else:
                st.success(f"Indexed {num_chunks} chunks from '{uploaded_file.name}'.")
                st.session_state.active_doc = namespace
                st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

documents = list_documents()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Settings")

    if not documents:
        st.warning("No documents indexed yet. Upload a PDF to start.")
        active_doc = None
    else:
        default_idx = 0
        if st.session_state.get("active_doc") in documents:
            default_idx = documents.index(st.session_state["active_doc"])
        active_doc = st.selectbox(
            "Active document",
            documents,
            index=default_idx,
            help="Questions are answered only from this document.",
        )
        st.session_state.active_doc = active_doc

    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("⚠️ Document Management")
    if active_doc and st.button(f"🗑️ Delete '{active_doc}'"):
        index.delete(delete_all=True, namespace=active_doc)
        st.session_state.messages = []
        st.session_state.pop("active_doc", None)
        st.success(f"Deleted '{active_doc}'.")
        st.rerun()

# ---------------- CHAT ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input(
    "Ask a question..." if active_doc else "Upload a PDF first...",
    disabled=not active_doc,
)

if user_input and active_doc:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, matches, search_query = ask(
                user_input, active_doc, st.session_state.messages[:-1]
            )
        st.markdown(answer)

        with st.expander(f"Sources used ({len(matches)})"):
            if search_query != user_input:
                st.caption(f"Searched for: _{search_query}_")
            for i, m in enumerate(matches, 1):
                meta = m.get("metadata", {}) or {}
                st.write(f"**[{i}]** page {meta.get('page', '?')} · score {m['score']:.3f}")
                st.write(meta.get("text", ""))
                st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})
