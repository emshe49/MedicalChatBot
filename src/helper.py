"""
Data Pipeline & Helper Utilities for Dental AI Chatbot.
Handles Data Ingestion, Preprocessing, Chunking, Embeddings, and Pinecone Indexing.
Supports both zero-cost local embeddings (FastEmbed / BAAI) and cloud embeddings.
"""

import os
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import dotenv
import pypdf
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# Windows symlink workaround for HuggingFace / FastEmbed
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Directory & Env setup
dotenv_path = Path(__file__).resolve().parent.parent / ".env"

# Cache fastembed model instance to avoid re-loading
_fastembed_instance = None


def get_config() -> Dict[str, str]:
    """
    Retrieve runtime configuration from project .env.
    Uses dotenv_values to ensure project .env takes precedence over stale OS environment variables.
    """
    env_vals = {}
    if dotenv_path.exists():
        env_vals = dotenv.dotenv_values(dotenv_path)

    def get_val(key: str, default: str = "") -> str:
        val = env_vals.get(key)
        if val is None or not str(val).strip():
            val = os.getenv(key, default)
        return str(val).strip().strip("'\"")

    return {
        "OPENROUTER_API_KEY": get_val("OPENROUTER_API_KEY", ""),
        "OPENROUTER_BASE_URL": get_val("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        "LLM_MODEL": get_val("LLM_MODEL", "openrouter/free"),
        "EMBEDDING_MODEL": get_val("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        "PINECONE_API_KEY": get_val("PINECONE_API_KEY", ""),
        "PINECONE_INDEX_NAME": get_val("PINECONE_INDEX_NAME", "dental-ortho-kb"),
        "PINECONE_CLOUD": get_val("PINECONE_CLOUD", "aws"),
        "PINECONE_REGION": get_val("PINECONE_REGION", "us-east-1"),
    }


# =====================================================================
# 1. DATA INGESTION
# =====================================================================
def load_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract raw text and page metadata from a PDF file using pypdf.
    Returns list of {'page': int, 'text': str, 'source': str}.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    filename = path.name
    reader = pypdf.PdfReader(str(path))
    pages_data = []

    for idx, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""
        pages_data.append({
            "page": idx + 1,
            "text": raw_text,
            "source": filename
        })

    return pages_data


# =====================================================================
# 2. PREPROCESSING
# =====================================================================
def clean_text(text: str) -> str:
    """
    Clean extracted text:
    - Normalizes line breaks and whitespace.
    - Fixes hyphenated line-breaks (e.g. 'ortho-\\ndontic' -> 'orthodontic').
    - Removes isolated standalone page numbers and non-printable control chars.
    """
    if not text:
        return ""

    # Replace windows line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Fix broken hyphenated words across lines
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

    # Replace multiple newlines with at most two
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Replace multiple horizontal spaces/tabs with single space
    text = re.sub(r'[ \t]+', ' ', text)

    # Strip standalone page numbering lines (e.g. "\n 12 \n")
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

    # Remove non-ascii non-printable characters while preserving standard punctuation
    text = "".join(ch for ch in text if ch.isprintable() or ch in ('\n', '\t'))

    return text.strip()


def preprocess_documents(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean all pages and discard empty pages."""
    cleaned = []
    for p in pages:
        cleaned_str = clean_text(p["text"])
        if len(cleaned_str) > 30:  # ignore blank or near-empty pages
            cleaned.append({
                "page": p["page"],
                "text": cleaned_str,
                "source": p["source"]
            })
    return cleaned


# =====================================================================
# 3. DATA CHUNKING
# =====================================================================
def split_into_chunks(
    documents: List[Dict[str, Any]], 
    chunk_size: int = 900, 
    chunk_overlap: int = 150
) -> List[Dict[str, Any]]:
    """
    Splits document pages into overlapping semantic text chunks.
    Ensures each chunk retains metadata (page, source, unique ID).
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "; ", ", ", " "]
        )
    except Exception:
        class SimpleSplitter:
            def __init__(self, size, overlap):
                self.size = size
                self.overlap = overlap
            def split_text(self, text):
                chunks = []
                start = 0
                while start < len(text):
                    end = start + self.size
                    chunks.append(text[start:end])
                    start = end - self.overlap
                return chunks
        splitter = SimpleSplitter(chunk_size, chunk_overlap)

    all_chunks = []
    global_chunk_idx = 0

    for doc in documents:
        page_num = doc["page"]
        source = doc["source"]
        page_text = doc["text"]

        text_pieces = splitter.split_text(page_text)
        for piece_idx, piece in enumerate(text_pieces):
            clean_piece = piece.strip()
            if len(clean_piece) < 25:
                continue

            chunk_id = f"{Path(source).stem}_p{page_num}_c{piece_idx}_{global_chunk_idx}"
            all_chunks.append({
                "id": chunk_id,
                "text": clean_piece,
                "page": page_num,
                "source": source,
                "chunk_idx": piece_idx
            })
            global_chunk_idx += 1

    return all_chunks


# =====================================================================
# 4. EMBEDDINGS (Local FastEmbed or OpenRouter)
# =====================================================================
def get_fastembed_model(model_name: str = "BAAI/bge-small-en-v1.5"):
    """Get or initialize the local FastEmbed model (0 cost, runs on CPU)."""
    global _fastembed_instance
    if _fastembed_instance is None:
        from fastembed import TextEmbedding
        _fastembed_instance = TextEmbedding(model_name=model_name)
    return _fastembed_instance


def get_embeddings_client() -> OpenAI:
    """Initialize OpenAI client configured for OpenRouter."""
    config = get_config()
    api_key = config["OPENROUTER_API_KEY"]
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set. Please provide it in .env or via Admin Settings.")

    return OpenAI(
        base_url=config["OPENROUTER_BASE_URL"],
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://localhost:5000",
            "X-Title": "Dental-RAG-Chatbot"
        }
    )


def generate_embeddings(
    texts: List[str], 
    batch_size: int = 32,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[List[float]]:
    """
    Generate vector embeddings for a list of text strings.
    Defaults to FastEmbed (zero-cost, no credits needed), with OpenRouter fallback.
    """
    config = get_config()
    model_name = config["EMBEDDING_MODEL"]

    # If model is FastEmbed (e.g. BAAI/bge-small-en-v1.5 or fastembed)
    if "bge" in model_name.lower() or "fastembed" in model_name.lower() or not config["OPENROUTER_API_KEY"]:
        embed_model = get_fastembed_model("BAAI/bge-small-en-v1.5")
        all_embeddings = []
        total = len(texts)
        
        # FastEmbed supports batch generator
        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            batch_vectors = [v.tolist() for v in embed_model.embed(batch)]
            all_embeddings.extend(batch_vectors)
            if progress_callback:
                progress_callback(min(i + batch_size, total), total)

        return all_embeddings

    # Otherwise attempt OpenRouter cloud embeddings
    try:
        client = get_embeddings_client()
        all_embeddings = []
        total = len(texts)

        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            batch = [t if t.strip() else "dental knowledge" for t in batch]

            response = client.embeddings.create(
                model=model_name,
                input=batch
            )
            sorted_data = sorted(response.data, key=lambda x: x.index)
            all_embeddings.extend([item.embedding for item in sorted_data])

            if progress_callback:
                progress_callback(min(i + batch_size, total), total)

        return all_embeddings

    except Exception as e:
        # If OpenRouter returns 402 or error, seamlessly fall back to local FastEmbed!
        print(f"Cloud embedding failed ({e}). Seamlessly switching to local FastEmbed BAAI model (0 credits required).")
        embed_model = get_fastembed_model("BAAI/bge-small-en-v1.5")
        return [v.tolist() for v in embed_model.embed(texts)]


def generate_query_embedding(query: str) -> List[float]:
    """Generate embedding for a single user query."""
    embeddings = generate_embeddings([query], batch_size=1)
    return embeddings[0]


# =====================================================================
# 5. PINECONE VECTOR DB OPERATIONS
# =====================================================================
def get_pinecone_client() -> Pinecone:
    """Initialize Pinecone client with project configuration."""
    config = get_config()
    api_key = config["PINECONE_API_KEY"]
    if not api_key:
        raise ValueError("PINECONE_API_KEY is not set. Please provide it in .env or via Admin Settings.")
    return Pinecone(api_key=api_key)


def get_or_create_pinecone_index(dimension: int = 384) -> Any:
    """
    Connect to existing Pinecone index or create a new Serverless index if absent.
    """
    config = get_config()
    index_name = config["PINECONE_INDEX_NAME"]
    pc = get_pinecone_client()

    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if index_name not in existing_indexes:
        cloud = config.get("PINECONE_CLOUD", "aws")
        region = config.get("PINECONE_REGION", "us-east-1")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=cloud,
                region=region
            )
        )
        time.sleep(3)

    return pc.Index(index_name)


def upsert_chunks_to_pinecone(
    chunks: List[Dict[str, Any]], 
    embeddings: List[List[float]],
    batch_size: int = 50,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> int:
    """
    Upsert chunks and embeddings into Pinecone with rich metadata.
    """
    if len(chunks) != len(embeddings):
        raise ValueError("Mismatch between number of chunks and embeddings.")

    dim = len(embeddings[0]) if embeddings else 384
    index = get_or_create_pinecone_index(dimension=dim)

    total = len(chunks)
    upserted_count = 0

    for i in range(0, total, batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]

        vectors_to_upsert = []
        for chunk, emb in zip(batch_chunks, batch_embeddings):
            meta_text = chunk["text"][:1500]
            vectors_to_upsert.append({
                "id": chunk["id"],
                "values": emb,
                "metadata": {
                    "text": meta_text,
                    "page": chunk["page"],
                    "source": chunk["source"],
                    "chunk_idx": chunk.get("chunk_idx", 0)
                }
            })

        index.upsert(vectors=vectors_to_upsert)
        upserted_count += len(vectors_to_upsert)

        if progress_callback:
            progress_callback(upserted_count, total)

    return upserted_count


def get_pinecone_stats() -> Dict[str, Any]:
    """Retrieve live index statistics from Pinecone."""
    try:
        config = get_config()
        if not config["PINECONE_API_KEY"]:
            return {"status": "not_configured", "vector_count": 0}

        pc = get_pinecone_client()
        index_name = config["PINECONE_INDEX_NAME"]
        existing = [idx.name for idx in pc.list_indexes()]

        if index_name not in existing:
            return {"status": "index_not_found", "index_name": index_name, "vector_count": 0}

        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        return {
            "status": "ready",
            "index_name": index_name,
            "vector_count": stats.get("total_vector_count", 0),
            "dimension": stats.get("dimension", 384),
            "namespaces": stats.get("namespaces", {})
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "vector_count": 0}


# =====================================================================
# 6. END-TO-END PIPELINE RUNNER
# =====================================================================
def run_full_ingestion_pipeline(
    pdf_path: str,
    progress_callback: Optional[Callable[[str, str, int, Dict[str, Any]], None]] = None
) -> Dict[str, Any]:
    """
    Executes the 5-stage Data Ingestion Pipeline:
      Stage 1: Data Ingestion (PDF parsing)
      Stage 2: Preprocessing (cleaning & normalization)
      Stage 3: Data Chunking (semantic overlapping text splitting)
      Stage 4: Data Embeddings (dense vector generation)
      Stage 5: Vector Indexing (Pinecone upsert)
    """
    def emit(stage: str, msg: str, pct: int, extra: Optional[Dict[str, Any]] = None):
        if progress_callback:
            progress_callback(stage, msg, pct, extra or {})

    # Stage 1: Data Ingestion
    emit("ingestion", f"Extracting pages from {Path(pdf_path).name}...", 10)
    raw_pages = load_pdf(pdf_path)
    total_pages = len(raw_pages)
    emit("ingestion", f"Successfully extracted {total_pages} pages.", 20, {"total_pages": total_pages})

    # Stage 2: Preprocessing
    emit("preprocessing", "Cleaning text, fixing line-breaks, and normalizing format...", 30)
    cleaned_pages = preprocess_documents(raw_pages)
    emit("preprocessing", f"Preprocessed {len(cleaned_pages)} non-empty clinical pages.", 40, {"cleaned_pages": len(cleaned_pages)})

    # Stage 3: Data Chunking
    emit("chunking", "Splitting pages into semantic overlapping chunks...", 50)
    chunks = split_into_chunks(cleaned_pages, chunk_size=950, chunk_overlap=150)
    total_chunks = len(chunks)
    emit("chunking", f"Generated {total_chunks} chunk segments with page metadata.", 60, {"total_chunks": total_chunks})

    # Stage 4: Data Embeddings
    emit("embeddings", f"Generating vector embeddings for {total_chunks} chunks...", 65)
    texts_to_embed = [c["text"] for c in chunks]

    def embedding_progress(done, total):
        pct = 65 + int((done / total) * 20)
        emit("embeddings", f"Generated embeddings: {done}/{total} chunks ({pct}%)...", pct)

    embeddings = generate_embeddings(texts_to_embed, batch_size=32, progress_callback=embedding_progress)
    emit("embeddings", f"Completed embedding generation for {len(embeddings)} vectors.", 85)

    # Stage 5: Pinecone Indexing
    emit("indexing", "Connecting to Pinecone and upserting vector index...", 88)

    def upsert_progress(done, total):
        pct = 88 + int((done / total) * 12)
        emit("indexing", f"Upserted to Pinecone: {done}/{total} vectors ({pct}%)...", pct)

    upserted = upsert_chunks_to_pinecone(chunks, embeddings, batch_size=50, progress_callback=upsert_progress)
    emit("complete", f"Pipeline completed! Successfully indexed {upserted} vectors into Pinecone.", 100, {
        "total_pages": total_pages,
        "total_chunks": total_chunks,
        "upserted_vectors": upserted
    })

    return {
        "status": "success",
        "total_pages": total_pages,
        "total_chunks": total_chunks,
        "upserted_vectors": upserted
    }
