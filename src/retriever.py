"""
Retrieval and Re-ranking Engine for Dental AI Chatbot.
Handles vector search on Pinecone, candidate re-ranking, and response synthesis via OpenRouter.
"""

import os
import re
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
from src.helper import (
    get_config, 
    generate_query_embedding, 
    get_or_create_pinecone_index
)
from src.prompt import (
    DENTIST_SYSTEM_PROMPT, 
    RAG_PROMPT_TEMPLATE
)

# Dedicated free chat models (excluding content moderation models)
FREE_MODEL_FALLBACKS = [
    "openrouter/free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "liquid/lfm-2.5-2.6b:free"
]


def get_llm_client() -> OpenAI:
    """Initialize OpenRouter LLM Client."""
    config = get_config()
    api_key = config["OPENROUTER_API_KEY"]
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not configured. Please add it in .env or via Admin Settings.")

    return OpenAI(
        base_url=config["OPENROUTER_BASE_URL"],
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://localhost:5000",
            "X-Title": "Dental-RAG-Chatbot"
        }
    )


# =====================================================================
# 1. PINECONE VECTOR RETRIEVAL
# =====================================================================
def retrieve_candidate_chunks(query: str, top_k: int = 12) -> List[Dict[str, Any]]:
    """
    Retrieve top_k semantic candidate chunks from Pinecone vector index.
    """
    query_emb = generate_query_embedding(query)
    index = get_or_create_pinecone_index(dimension=len(query_emb))

    results = index.query(
        vector=query_emb,
        top_k=top_k,
        include_metadata=True
    )

    candidates = []
    for match in results.get("matches", []):
        metadata = match.get("metadata", {})
        candidates.append({
            "id": match.get("id"),
            "score": float(match.get("score", 0.0)),
            "text": metadata.get("text", ""),
            "page": metadata.get("page", 0),
            "source": metadata.get("source", "Unknown Document")
        })

    return candidates


# =====================================================================
# 2. RE-RANKING ENGINE
# =====================================================================
def compute_lexical_overlap_score(query: str, text: str) -> float:
    """Calculate keyword overlap between patient query and clinical excerpt."""
    query_tokens = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))
    if not query_tokens:
        return 0.0
    text_lower = text.lower()
    matches = sum(1 for token in query_tokens if token in text_lower)
    return matches / len(query_tokens)


def rerank_chunks(
    query: str, 
    candidates: List[Dict[str, Any]], 
    top_n: int = 4
) -> List[Dict[str, Any]]:
    """
    Re-rank vector search candidates to pick the top_n most relevant clinical excerpts.
    Combines dense cosine similarity score with lexical terminology density.
    """
    if not candidates:
        return []

    if len(candidates) <= top_n:
        return sorted(candidates, key=lambda c: c["score"], reverse=True)

    for c in candidates:
        lex_score = compute_lexical_overlap_score(query, c["text"])
        # Blended rank score: 70% semantic vector similarity + 30% lexical density
        c["rerank_score"] = round(((c["score"] * 0.7) + (lex_score * 0.3)) * 10, 2)

    sorted_candidates = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return sorted_candidates[:top_n]


# =====================================================================
# 3. RESPONSE SYNTHESIS (DENTIST AI PERSONA)
# =====================================================================
def format_context_block(chunks: List[Dict[str, Any]]) -> str:
    """Format re-ranked chunks into clean prompt context."""
    if not chunks:
        return "No specific literature excerpts found. Provide expert guidance using general dental clinical knowledge."

    blocks = []
    for idx, c in enumerate(chunks):
        blocks.append(
            f"--- [EXCERPT {idx + 1} | Source: {c['source']} | Page: {c['page']}] ---\n{c['text']}"
        )
    return "\n\n".join(blocks)


def ask_dentist(
    query: str, 
    chat_history: Optional[List[Dict[str, str]]] = None,
    top_k: int = 12,
    top_n: int = 4
) -> Dict[str, Any]:
    """
    End-to-End Dental Chatbot Interaction:
    1. Query embedding & Pinecone retrieval (top_k).
    2. Candidate re-ranking (top_n).
    3. OpenRouter LLM answer generation with Dentist persona (with auto free-model fallback).
    4. Structured source attribution.
    """
    config = get_config()
    client = get_llm_client()
    target_model = config["LLM_MODEL"] or "openrouter/free"

    retrieved_chunks = []
    reranked_chunks = []
    context_str = ""

    # Attempt retrieval from Pinecone
    if config["PINECONE_API_KEY"]:
        try:
            retrieved_chunks = retrieve_candidate_chunks(query, top_k=top_k)
            reranked_chunks = rerank_chunks(query, retrieved_chunks, top_n=top_n)
            context_str = format_context_block(reranked_chunks)
        except Exception as e:
            print(f"Pinecone retrieval notice: {e}")
            context_str = "Pinecone retrieval unavailable or index empty. Answer using standard certified dental principles."

    # Build prompt
    prompt_content = RAG_PROMPT_TEMPLATE.format(
        context=context_str if context_str else "General Dental Clinical Knowledge",
        question=query
    )

    messages = [
        {"role": "system", "content": DENTIST_SYSTEM_PROMPT}
    ]

    # Append recent conversation history (up to last 6 turns)
    if chat_history:
        for turn in chat_history[-6:]:
            role = "user" if turn.get("role") in ("user", "patient") else "assistant"
            messages.append({
                "role": role,
                "content": turn.get("content", "")
            })

    messages.append({"role": "user", "content": prompt_content})

    # Prepare candidate models to try (primary + free fallbacks)
    models_to_attempt = [target_model]
    for fb in FREE_MODEL_FALLBACKS:
        if fb not in models_to_attempt:
            models_to_attempt.append(fb)

    answer_text = ""
    model_used = target_model
    last_error = None

    for model in models_to_attempt:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=2500
            )
            content = response.choices[0].message.content
            if content and content.strip():
                answer_text = content
                model_used = model
                break
        except Exception as e:
            last_error = e
            print(f"Model {model} failed ({e}), attempting next fallback...")

    if not answer_text:
        raise RuntimeError(f"All LLM models failed to generate response. Last error: {last_error}")

    # Build unique source citations
    sources = []
    seen_pages = set()
    for c in reranked_chunks:
        key = (c["source"], c["page"])
        if key not in seen_pages:
            seen_pages.add(key)
            sources.append({
                "source": c["source"],
                "page": c["page"],
                "snippet": c["text"][:180] + "...",
                "score": round(c.get("rerank_score", c.get("score", 0.0)), 2)
            })

    return {
        "answer": answer_text,
        "sources": sources,
        "retrieved_count": len(retrieved_chunks),
        "reranked_count": len(reranked_chunks),
        "model_used": model_used
    }


def stream_dentist_response(
    query: str, 
    chat_history: Optional[List[Dict[str, str]]] = None,
    top_k: int = 12,
    top_n: int = 4
):
    """
    Stream token-by-token Dental Chatbot Response:
    1. Retrieval & Re-ranking.
    2. Yields 'meta' event with source citations.
    3. Calls OpenRouter with stream=True and yields each 'token'.
    4. Yields 'done' event upon completion.
    """
    config = get_config()
    client = get_llm_client()
    target_model = config["LLM_MODEL"] or "openrouter/free"

    retrieved_chunks = []
    reranked_chunks = []
    context_str = ""

    # Check if query is simple conversational/identity rather than textbook dental question
    is_conversational = bool(re.search(r'\b(my name|your name|who are you|hello|hi\b|hey\b|how are you|remember me)\b', query.lower()))

    # 1. Retrieve & Rerank from Pinecone (only if query is clinical/topic-specific)
    if not is_conversational and config["PINECONE_API_KEY"]:
        try:
            retrieved_chunks = retrieve_candidate_chunks(query, top_k=top_k)
            reranked_chunks = rerank_chunks(query, retrieved_chunks, top_n=top_n)
            context_str = format_context_block(reranked_chunks)
        except Exception as e:
            print(f"Pinecone retrieval notice: {e}")
            context_str = "Pinecone retrieval unavailable or index empty. Answer using standard certified dental principles."

    # Build unique source citations
    sources = []
    seen_pages = set()
    for c in reranked_chunks:
        key = (c["source"], c["page"])
        if key not in seen_pages:
            seen_pages.add(key)
            sources.append({
                "source": c["source"],
                "page": c["page"],
                "snippet": c["text"][:180] + "...",
                "score": round(c.get("rerank_score", c.get("score", 0.0)), 2)
            })

    # Emit metadata event first
    yield f"data: {json.dumps({'type': 'meta', 'sources': sources, 'model_used': target_model})}\n\n"

    # 2. Build multi-turn memory messages
    if context_str:
        prompt_content = RAG_PROMPT_TEMPLATE.format(
            context=context_str,
            question=query
        )
    else:
        prompt_content = query

    messages = [
        {"role": "system", "content": DENTIST_SYSTEM_PROMPT}
    ]

    if chat_history:
        for turn in chat_history[-8:]:
            role = "user" if turn.get("role") in ("user", "patient") else "assistant"
            content = turn.get("content", "")
            if content:
                messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": prompt_content})

    # 3. Stream from model (with fallback)
    models_to_attempt = [target_model]
    for fb in FREE_MODEL_FALLBACKS:
        if fb not in models_to_attempt:
            models_to_attempt.append(fb)

    stream_success = False
    for model in models_to_attempt:
        try:
            stream_resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=2500,
                stream=True
            )

            buffered_tokens = []
            is_valid_chat = True

            for chunk in stream_resp:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        buffered_tokens.append(delta)
                        full_preview = "".join(buffered_tokens)

                        # Filter out safety classifier models like nvidia/nemotron-3.5-content-safety
                        if "User Safety: safe" in full_preview or "Response Safety:" in full_preview:
                            print(f"Skipping safety classifier output from {model}...")
                            is_valid_chat = False
                            break

                        yield f"data: {json.dumps({'type': 'token', 'token': delta})}\n\n"

            if is_valid_chat and len(buffered_tokens) > 0:
                stream_success = True
                break
        except Exception as e:
            print(f"Streaming with {model} failed ({e}), trying fallback...")

    if not stream_success:
        yield f"data: {json.dumps({'type': 'error', 'error': 'Unable to stream response from free LLM endpoints.'})}\n\n"
    else:
        yield f"data: {json.dumps({'type': 'done'})}\n\n"


