import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from markitdown import MarkItDown
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# ---------------------------------------------------------------------------
# Load API Key
# ---------------------------------------------------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Add it to your .env file.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DOCUMENT_FOLDER = "documents"
os.makedirs(DOCUMENT_FOLDER, exist_ok=True)

# ---------------------------------------------------------------------------
# Core Pipeline Functions
# ---------------------------------------------------------------------------
def load_documents():
    documents = []
    md_converter = MarkItDown()
    for file_name in os.listdir(DOCUMENT_FOLDER):
        file_path = os.path.join(DOCUMENT_FOLDER, file_name)
        text = ""
        if file_name.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        elif file_name.endswith(".pdf"):
            try:
                result = md_converter.convert(file_path)
                text = result.text_content
            except Exception:
                reader = PdfReader(file_path)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
        if text.strip():
            documents.append({"filename": file_name, "text": text})
    return documents


def split_documents(documents, chunk_size=500, overlap=100):
    chunks = []
    step = chunk_size - overlap
    for document in documents:
        text = document["text"]
        source = document["filename"]
        for i in range(0, len(text), step):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append({"content": chunk, "source": source})
    return chunks


def create_vectors(chunks):
    texts = [chunk["content"] for chunk in chunks]
    vectorizer = TfidfVectorizer(sublinear_tf=True)
    chunk_vectors = vectorizer.fit_transform(texts)
    return vectorizer, chunk_vectors


def retrieve_chunks(query, chunks, vectorizer, chunk_vectors, top_k=3):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
    ranked_indices = similarities.argsort()[::-1]
    threshold = 0.10
    best_per_doc = {}
    for idc in ranked_indices:
        score = similarities[idc]
        if score < threshold:
            break
        chunk = chunks[idc]
        source = chunk["source"]
        if source not in best_per_doc:
            best_per_doc[source] = {"content": chunk["content"], "score": score, "source": source}
    selected = sorted(best_per_doc.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    return selected


def decide_action(query):
    calculation_keywords = ["calculate", "compute", "+", "*", "/"]
    query_lower = query.lower()
    for keyword in calculation_keywords:
        if keyword in query_lower:
            return "calculation"
    return "document_search"


def calculate(query):
    expression = query.lower().replace("calculate", "").replace("compute", "").strip()
    try:
        result = eval(expression)
        return f"Result: {result}", "Calculator"
    except:
        return "Could not evaluate the expression.", "Calculator"


pronouns = {"he", "she", "it", "they", "his", "her", "their"}


def resolve_query(query, chat_history):
    words = query.lower().split()
    if chat_history and any(word in pronouns for word in words):
        last_answer = chat_history[-1]["answer"]
        first_sentence = last_answer.split(".")[0].strip()
        return first_sentence + " " + query
    return query


def answer_question(query, chunks, vectorizer, chunk_vectors, chat_history):
    resolved_query = resolve_query(query, chat_history)
    relevant_chunks = retrieve_chunks(resolved_query, chunks, vectorizer, chunk_vectors)

    if not relevant_chunks:
        return "I couldn't find relevant information in the documents.", "N/A"

    context = "\n\n".join([f"Source: {c['source']}\n{c['content']}" for c in relevant_chunks])

    prompt = f"""You are a helpful assistant answering questions from a document.

If the context contains an MCQ question matching the user's query:
- Return the correct answer option with its text

If the context does not contain a matching MCQ but has relevant information:
- Answer descriptively based on the context

If the context has no relevant information at all:
- Say "I couldn't find relevant information in the documents."

IMPORTANT: Do NOT include SOURCE: inside the ANSWER field. Only use the exact format below.

Respond in this exact format:
ANSWER: <your answer>
SOURCE: <source filename>

Context:
{context}

Question: {resolved_query}"""

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = completion.choices[0].message.content.strip()
    answer = "No answer found."
    source = "Unknown"
    for line in raw.splitlines():
        if line.startswith("ANSWER:"):
            answer = line[len("ANSWER:"):].strip()
        elif line.startswith("SOURCE:"):
            source = line[len("SOURCE:"):].strip()
    return answer, source

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Agentic RAG", page_icon="🤖", layout="wide")
# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False
st.title("🤖 Agentic RAG with Groq")
st.caption("Upload documents, ask questions, get answers powered by LLaMA 3.1")

# ---------------------------------------------------------------------------
# Sidebar — File Upload + Load Documents
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📂 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            save_path = os.path.join(DOCUMENT_FOLDER, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"{len(uploaded_files)} file(s) saved to documents/")

    if st.button("📥 Load Documents", use_container_width=True):
        with st.spinner("Loading and indexing documents..."):
            documents = load_documents()
            if not documents:
                st.error("No documents found in the documents/ folder.")
            else:
                chunks = split_documents(documents)
                vectorizer, chunk_vectors = create_vectors(chunks)
                st.session_state.chunks = chunks
                st.session_state.vectorizer = vectorizer
                st.session_state.chunk_vectors = chunk_vectors
                st.session_state.documents_loaded = True
                st.success(f"Loaded {len(documents)} document(s), {len(chunks)} chunks indexed.")

    st.divider()

    # Show loaded files
    if os.path.exists(DOCUMENT_FOLDER):
        files = [f for f in os.listdir(DOCUMENT_FOLDER) if f.endswith((".pdf", ".txt"))]
        if files:
            st.markdown("**Files in documents/:**")
            for f in files:
                st.markdown(f"- {f}")

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ---------------------------------------------------------------------------
# Chat Display
# ---------------------------------------------------------------------------
chat_container = st.container()

with chat_container:
    if not st.session_state.chat_history:
        st.info("Load documents from the sidebar, then ask a question below.")
    else:
        for entry in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(entry["question"])
            with st.chat_message("assistant"):
                st.write(entry["answer"])
                if entry["source"] not in ("N/A", "Unknown", "Calculator"):
                    st.caption(f"Source: {entry['source']}")

# ---------------------------------------------------------------------------
# Chat Input
# ---------------------------------------------------------------------------
query = st.chat_input("Ask a question...")

if query:
    if not st.session_state.documents_loaded:
        st.warning("Please load documents first using the sidebar.")
    else:
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                action = decide_action(query)
                if action == "calculation":
                    answer, source = calculate(query)
                else:
                    answer, source = answer_question(
                        query,
                        st.session_state.chunks,
                        st.session_state.vectorizer,
                        st.session_state.chunk_vectors,
                        st.session_state.chat_history
                    )
            st.write(answer)
            if source not in ("N/A", "Unknown", "Calculator"):
                st.caption(f"Source: {source}")

        st.session_state.chat_history.append({
            "question": query,
            "answer": answer,
            "source": source
        })