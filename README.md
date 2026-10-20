# AI TechCorp Employee Assistant (RAG Pipeline)

A Retrieval-Augmented Generation (RAG) assistant designed to help employees quickly find accurate answers to questions regarding company policies, benefits, and procedures based on the official employee handbook.

The project features a full LangChain RAG pipeline powered by Google Gemini models, persistent vector storage with Chroma DB, and offers both an interactive **Streamlit web chat interface** and a **terminal-based CLI**.

---

## Features

- **Document Grounding**: Answers are strictly grounded on the official company handbook (`TechCorp_Official_Employee_Handbook.pdf`), preventing hallucinations.
- **Dual Interface**:
  - **Streamlit Web UI**: Clean, interactive chat UI with message bubbles and loading indicators.
  - **Terminal CLI**: Fast command-line loop for instant terminal testing and Q&A.
- **Chroma Vector Store**: Document embeddings are indexed and persisted locally using Chroma DB.
- **Powered by Google Gemini**: Utilizes `gemini-embedding-001` for vector representations and `gemini-3.5-flash-lite` for concise, deterministic, and fast responses.
- **LangChain Expression Language (LCEL)**: Modular and composable pipeline combining retrieval, formatting, prompt engineering, and LLM generation.

---

## Architecture & Tech Stack

```text
[ TechCorp Handbook (PDF) ]
            │
            ▼
   [ PyPDFLoader ]
            │
            ▼
[ RecursiveCharacterTextSplitter ] (Chunk Size: 500, Overlap: 50)
            │
            ▼
 [ Google Generative AI Embeddings ] (models/gemini-embedding-001)
            │
            ▼
    [ Chroma Vector Store ] ──(Top k=2 Documents)──┐
                                                    ▼
[ User Question ] ─────────────────────────► [ Prompt Template ]
                                                    │
                                                    ▼
                                    [ ChatGoogleGenerativeAI ] (gemini-3.5-flash-lite)
                                                    │
                                                    ▼
                                      [ Response / Streamlit UI ]
```

### Key Libraries
- **[LangChain](https://python.langchain.com/)**: Document loading, text splitting, LCEL chaining, and retrieval.
- **[LangChain Google GenAI](https://github.com/langchain-ai/langchain-google)**: Embeddings (`gemini-embedding-001`) and LLM (`gemini-3.5-flash-lite`).
- **[Chroma DB](https://www.trychroma.com/)**: Local vector database for embedding indexing and similarity search.
- **[PyPDF](https://pypdf.readthedocs.io/)**: PDF document extraction.
- **[Streamlit](https://streamlit.io/)**: Interactive web application interface.
- **[python-dotenv](https://github.com/theskumar/python-dotenv)**: Secure API key configuration.

---

## Project Structure

```text
my-rag-project/
├── TechCorp_Official_Employee_Handbook.pdf  # Source handbook document
├── rag_app.py                               # Core RAG pipeline, embeddings, Chroma DB setup & CLI
├── app.py                                   # Streamlit chat web application
├── requirements.txt                         # Python dependencies
├── chroma_db/                               # Local persistent vector store
├── .env                                     # Environment variables (API keys - excluded from Git)
└── README.md                                # Project documentation
```

---

## Prerequisites & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Tom-GP/my-rag-project.git
cd my-rag-project
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Key
Create a `.env` file in the root directory and add your Google Gemini API key:
```env
GEMINI-API-KEY=your_gemini_api_key_here
```
> **Note**: Obtain an API key from [Google AI Studio](https://aistudio.google.com/).

---

## Usage

### Option 1: Run the Streamlit Web Application
Launch the interactive web UI:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to start chatting with the assistant.

### Option 2: Run in Terminal (CLI Mode)
You can query the RAG model directly in your command line:
```bash
python rag_app.py
```
Type your question when prompted (e.g., *"What days can I work from home?"*). Type `exit` or `quit` to end the session.

---

## How It Works

1. **Document Loading**: `rag_app.py` reads `TechCorp_Official_Employee_Handbook.pdf` via `PyPDFLoader`.
2. **Chunking**: Text is split into 500-character segments with a 50-character overlap using `RecursiveCharacterTextSplitter`.
3. **Embedding & Storage**: Document chunks are vectorized using `GoogleGenerativeAIEmbeddings` and stored in `chroma_db/`.
4. **Retrieval**: When a query is submitted, the retriever selects the 2 most relevant chunks based on semantic similarity.
5. **Generation**: The context chunks and user question are injected into the prompt and synthesized into a concise answer by `gemini-3.5-flash-lite`.
