# Title: Agentic RAG with Groq

A command-line and web-based Retrieval-Augmented Generation (RAG) system built using Python, Groq's LLaMA 3.1 model, TF-IDF retrieval,Streamlit UI and an agentic decision layer. The system loads documents, chunks and indexes them, and answers user queries by retrieving relevant context and generating accurate responses.

---

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)
![Groq](https://img.shields.io/badge/Groq-LLaMA3.1-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF-blue?logo=scikit-learn)
![MarkItDown](https://img.shields.io/badge/MarkItDown-Microsoft-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Abstract

This project implements a document-based question answering system using the Retrieval-Augmented Generation (RAG) architecture. It is built with Python and powered by Groq's LLaMA 3.1 8B model.

The system loads PDF and TXT documents from a local folder, using MarkItDown for high-quality PDF-to-Markdown conversion with pypdf as fallback... splits them into overlapping chunks, and indexes them using TF-IDF vectorization. When a user submits a query, the system computes cosine similarity between the query vector and all chunk vectors to retrieve the most relevant context. An agentic decision layer routes the query — either to the document retrieval pipeline or to a calculator tool based on the nature of the query. A pronoun resolution module resolves contextual references using chat history before retrieval. The retrieved context and query are then passed to Groq's LLaMA 3.1 model, which generates a structured response with an answer and source citation.

The system preserves conversational continuity through a chat history buffer, enabling multi-turn interactions. API credentials are managed securely using environment variables.

This project demonstrates core RAG concepts including document chunking, TF-IDF retrieval, cosine similarity ranking, agentic routing, pronoun resolution, and LLM API integration in a modular Python pipeline.

---

## Key Concepts Demonstrated

- Retrieval-Augmented Generation (RAG)
- Document Chunking with Overlap
- TF-IDF Vectorization
- Cosine Similarity Ranking
- Agentic Decision Making
- Pronoun Resolution and Query Rewriting
- LLM API Integration
- Conversational Context Preservation
- Multi-document Source Attribution
- Environment Variable Management
- MarkItDown-powered Document Preprocessing

---

## System Workflow

```text
┌──────────────────────┐
│   Document Loading   │
│  (MarkItDown + pypdf)│
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│     User Query       │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│   Pronoun Resolver   │
│  (resolve_query)     │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│   Agent Decision     │
│  (decide_action)     │
└────┬─────────────────┘
     │
     ├──── calculation ──→ ┌──────────────────┐
     │                     │  Calculator Tool │
     │                     └──────────────────┘
     │
     └── document_search →  ┌──────────────────────┐
                            │   TF-IDF Retrieval   │
                            │  (retrieve_chunks)   │
                            └──────────┬───────────┘
                                       ↓
                            ┌──────────────────────┐
                            │  Groq API (LLaMA 3.1)│
                            └──────────┬───────────┘
                                       ↓
                            ┌──────────────────────┐
                            │   Answer + Source    │
                            └──────────────────────┘
```

---

## Project Structure

```text
agentic-rag-groq/
│
├── main.py
├── app.py
├── requirements.txt
├── complete_architecture.txt
│
├── documents/              ← Add your PDF/TXT files here (not tracked by git)
│
├── LICENSES/
│   └── markitdown-MIT.txt
│
├── .env
├── .gitignore
├── LICENSE
└── README.md
```

### File Description

- **main.py** — Core pipeline: document loading, chunking, TF-IDF indexing, retrieval, agentic routing, pronoun resolution, and Groq LLM integration.
- **app.py** — Streamlit web interface with file uploader, chat history display, and source attribution.
- **requirements.txt** — Python dependencies.
- **complete_architecture.txt** — Detailed system architecture notes.
- **documents/** — Folder for user-provided PDF and TXT files. Not tracked by git.
- **LICENSES/markitdown-MIT.txt** — Third-party license for Microsoft MarkItDown.
- **.env** — Stores the Groq API key securely.
- **LICENSE** — MIT License for this project.

---

## Technologies Used

### Frontend
- Streamlit

### Backend
- Python
- Groq API
- Python Dotenv
- pypdf
- scikit-learn (TF-IDF, Cosine Similarity)
- MarkItDown (Microsoft) — PDF to Markdown conversion

### AI Model
- LLaMA 3.1 8B Instant (via Groq)

---

## Features

- Multi-document loading (PDF and TXT)
- Overlapping document chunking for context preservation
- TF-IDF vectorization and cosine similarity retrieval
- Agentic routing — document search or calculator based on query type
- Pronoun resolution using chat history
- Best chunk selection per document to avoid redundancy
- Relevance threshold filtering
- Source attribution in every response
- Multi-turn conversational context via chat history buffer
- Secure API key management using environment variables
- MarkItDown-powered PDF extraction for cleaner text quality
- Streamlit web interface with chat history and file uploader

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/Blackblitz777/agentic-rag-groq.git
cd agentic-rag-groq
```

### Create a Virtual Environment

```bash
python -m venv venv
```

### Activate the Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

Linux / WSL:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY="your_api_key_here"
```

### Add Documents

Create a `documents/` folder and add your PDF or TXT files:

```bash
mkdir documents
cp /path/to/yourfile.pdf documents/
```

### Run the Application

**CLI:**
```bash
python main.py
```
**Web UI:**
```bash
streamlit run app.py
```
---

## Usage

### CLI
1. Place your PDF or TXT files in the `documents/` folder.
2. Run `python main.py`.
3. The system loads, chunks, and indexes all documents.
4. Type your question at the prompt.
5. The agent decides whether to search documents or calculate.
6. Relevant chunks are retrieved and passed to LLaMA 3.1.
7. The answer and source filename are displayed.
8. Type `quit` or `exit` to stop.

### Web UI (Streamlit)
1. Run `streamlit run app.py`.
2. Upload PDFs via the sidebar or place them in the `documents/` folder.
3. Click **Load Documents** to index them.
4. Ask questions in the chat input.
5. The agent decides whether to search documents or calculate.
6. Relevant chunks are retrieved and passed to LLaMA 3.1.
7. The answer and source are displayed in the chat.
8. Use **Clear Chat** in the sidebar to reset the conversation.

---

## Learning Outcomes

This project helped demonstrate:

- RAG Pipeline Architecture
- Document Chunking and Overlap Strategy
- TF-IDF Vectorization and Cosine Similarity
- Agentic Decision Making
- Pronoun Resolution and Query Rewriting
- LLM API Integration with Groq
- Multi-document Retrieval and Source Attribution
- Conversational Memory Handling
- Python Modular Pipeline Design
- PDF to Markdown conversion using MarkItDown

---

## Future Improvements

- Replace TF-IDF with semantic embeddings (FAISS + sentence-transformers)
- Persistent chat history using a database
- Streaming responses
- Multi-model selection
- Reranking retrieved chunks before LLM call
- Deployment using Docker and cloud platforms

---
## Design Decisions

### Why TF-IDF instead of Dense Embeddings?

This implementation intentionally uses TF-IDF over dense embeddings (FAISS + sentence-transformers) to demonstrate the classical retrieval stage of a RAG pipeline.

**Advantages:**
- Fast and lightweight — no embedding model required
- No GPU needed
- Easy to understand and debug
- Sufficient for keyword-heavy documents like question banks

**Known Limitation:**
- Semantic queries ("What is a transformer model?") may retrieve wrong chunks since TF-IDF matches keywords, not meaning
- Future versions will migrate to FAISS + sentence-transformers for semantic search

---

## Third-Party Licenses

- [MarkItDown](https://github.com/microsoft/markitdown) by Microsoft — MIT License. See `LICENSES/markitdown-MIT.txt`.

---

## Author

Developed as part of Generative AI and Large Language Model learning and experimentation.
