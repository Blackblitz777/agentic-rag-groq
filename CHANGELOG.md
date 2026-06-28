# Changelog

All notable changes to this project are documented here.

---

## v1.3 — 2026-06-28

### Added
- Streamlit web interface (`app.py`) with sidebar file uploader, chat history display, and source attribution
- Load Documents button to manually trigger indexing
- Clear Chat button to reset conversation in the UI
- Session state management for persistent chat across Streamlit reruns

### Changed
- Prompt updated to handle both MCQ and descriptive queries intelligently
- Empty query now skipped in CLI loop instead of being processed

---

## v1.2 — 2026-06-27

### Added
- MarkItDown integration for high-quality PDF-to-Markdown conversion
- pypdf retained as fallback if MarkItDown fails
- `LICENSES/markitdown-MIT.txt` — third-party license for Microsoft MarkItDown
- `source` key added to `best_per_doc` dict in `retrieve_chunks` to fix KeyError

### Changed
- `decide_action` keywords cleaned up — removed `-`, `find`, `solve` to prevent false routing on queries like "What is fine-tuning?"
- `load_documents` fallback pypdf now correctly indented inside `except` block

---

## v1.1 — 2026-06-26

### Added
- Agentic decision layer (`decide_action`) routing queries to calculator or document search
- Calculator tool using `eval()` with error handling
- Pronoun resolution (`resolve_query`) using chat history to handle follow-up queries
- Relevance threshold filtering (0.10) in `retrieve_chunks`
- Best chunk per document selection to avoid redundancy
- Chat history buffer for multi-turn conversation context
- `source` attribution in every LLM response
- `.gitignore` configured for venv, .env, pycache, documents folder
- MIT License and third-party license structure

### Fixed
- Key mismatch: `document['file_name']` → `document['filename']`
- Infinite recursion in `is_pronoun_query` — rewritten as `resolve_query`
- Floating Groq API call wrapped into `answer_question()` function
- Chat loop indentation — action/answer/print now correctly inside `while` block
- `answer` and `source` variables initialized before for loop to prevent `UnboundLocalError`

---

## v1.0 — 2026-06-26

### Added
- Initial RAG pipeline (`main.py`)
- PDF and TXT document loading using pypdf
- Overlapping document chunking (chunk_size=500, overlap=100)
- TF-IDF vectorization using scikit-learn
- Cosine similarity retrieval
- Groq LLaMA 3.1 8B Instant integration
- CLI chat loop with quit/exit support
- Environment variable management via python-dotenv