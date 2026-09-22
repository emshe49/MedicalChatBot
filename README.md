# 🦷 Dr. Pearl | Dental AI Chatbot & Knowledge Hub

An intelligent, clinically-grounded Dental & Orthodontic AI Assistant backed by **RAG (Retrieval-Augmented Generation)**, **Pinecone Vector Database**, **OpenRouter API**, and **Re-ranking**. Features both a patient-facing **Chat Interface** and a dedicated **Admin Knowledge Management Dashboard**.

---

## 🌟 Key Features

1. **Dentist & Orthodontist AI Persona**:
   - Actively counsels patients with empathy, clinical accuracy, and dental hygiene protocols.
   - Tailored specifically to orthodontic conditions (crowding, spacing, malocclusion, braces, aligners, tooth sensitivity, oral hygiene).
   - Clinical safety triage and emergency disclaimers.

2. **Full-Stack 5-Stage Ingestion Pipeline**:
   - **Stage 1: Data Ingestion**: Parses PDFs (including the pre-loaded 130-page *Guide to Orthodontics* textbook).
   - **Stage 2: Preprocessing**: Normalizes line breaks, eliminates hyphenated page splits, strips noise artifacts.
   - **Stage 3: Data Chunking**: Recursive semantic text splitting with page & document metadata.
   - **Stage 4: Embeddings**: Generates dense vector embeddings via OpenRouter API (OpenAI-compatible `text-embedding-3-small`).
   - **Stage 5: Pinecone Indexing**: Upserts vectors and metadata into Pinecone Serverless Vector DB.

3. **Advanced Retrieval & Re-ranking**:
   - Dense semantic vector search retrieves candidate excerpts (`top_k=12`).
   - Multi-factor Re-ranking scores lexical and clinical relevance, selecting top-4 high-precision passages to inject into the LLM context.

4. **Dual Interfaces**:
   - **Patient Chat UI (`/`)**: Modern medical aesthetic, suggested prompts, speech-to-text voice input, verified source citations viewer, copy answers.
   - **Admin Ingestion Hub (`/admin`)**: Drag-and-drop PDF textbook upload, 5-stage visual stepper, live streaming console logs, vector statistics, and API key manager.

---

## 🚀 Quick Start

### 1. Install Dependencies
Ensure you have Python installed:
```bash
pip install -r MedicalChatBot/requirements.txt
```

### 2. Configure API Keys
You can configure your keys either in `MedicalChatBot/.env` or directly through the **Admin Settings UI** (`/admin`):
```ini
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=meta-llama/llama-3.3-70b-instruct
EMBEDDING_MODEL=text-embedding-3-small

PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=dental-ortho-kb
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

### 3. Launch the Application
Run from the project root:
```bash
python run.py
```
Or from inside `MedicalChatBot/`:
```bash
python app.py
```

### 4. Access the Interfaces
- **Patient Chatbot**: [http://localhost:5000/](http://localhost:5000/)
- **Admin Ingestion Dashboard**: [http://localhost:5000/admin](http://localhost:5000/admin)

---

## 📂 Project Architecture

```
MedicalChatbot/
├── run.py                       # Root launcher script
└── MedicalChatBot/
    ├── app.py                   # Flask server & REST endpoints
    ├── requirements.txt         # Project dependencies
    ├── .env                     # Local environment variables
    ├── .env.example             # Environment template
    ├── src/
    │   ├── __init__.py
    │   ├── a_20guide_20to_20orthodontics.pdf  # Stored 130-page Orthodontics textbook
    │   ├── prompt.py            # Clinical Dentist persona & RAG prompts
    │   ├── helper.py            # PDF loader, preprocessing, chunker, embeddings & Pinecone upsert
    │   └── retriever.py         # Vector search, re-ranking & synthesis
    ├── templates/
    │   ├── index.html           # Patient Dental Chat UI
    │   └── admin.html           # Admin Knowledge Management Dashboard
    └── static/
        ├── css/
        │   └── style.css        # Clean medical dark & teal design system
        └── js/
            ├── chat.js          # Patient chat client & citations modal
            └── admin.js         # Admin stepper, uploader, & live terminal logger
```
