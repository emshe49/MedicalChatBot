"""
Flask Application for Dental AI Chatbot & Knowledge Management Admin Dashboard.
Provides patient chat interface and administrator PDF ingestion pipeline.
"""

import os
import time
import json
import threading
from pathlib import Path
from functools import wraps
from flask import Flask, render_template, request, jsonify, Response, session, redirect, url_for
from flask_cors import CORS
from dotenv import set_key

from seed_admin import verify_admin_login, seed_admin
from src.helper import (
    get_config,
    get_pinecone_stats,
    load_pdf,
    run_full_ingestion_pipeline,
    dotenv_path
)
from src.retriever import ask_dentist, stream_dentist_response

# Directory paths
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

app = Flask(
    __name__,
    template_folder=str(TEMPLATES_DIR),
    static_folder=str(STATIC_DIR)
)
app.secret_key = "dental-ai-chatbot-secret-key-2026-auth"
CORS(app)

# Ensure default admin exists in SQLite
seed_admin()

# Admin authentication decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("admin_logged_in"):
            if request.path.startswith("/api/admin/"):
                return jsonify({"success": False, "error": "Unauthorized. Please log in as admin."}), 401
            return redirect(url_for("login", next=request.path))
        return f(*args, **kwargs)
    return decorated_function

# In-memory pipeline status tracker
pipeline_state = {
    "is_running": False,
    "current_stage": "idle",
    "percentage": 0,
    "message": "Ready to run pipeline",
    "logs": [],
    "result": None,
    "error": None
}


# =====================================================================
# UI ROUTES & AUTH
# =====================================================================
@app.route("/")
def index():
    """User-facing Dental AI Chatbot Interface."""
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    """Administrator Login Portal."""
    if session.get("admin_logged_in"):
        return redirect(url_for("admin"))

    error = None
    email_val = ""
    if request.method == "POST":
        email_val = request.form.get("email", "").strip()
        password_val = request.form.get("password", "").strip()

        if not email_val or not password_val:
            error = "Please enter both administrator email and password."
        elif verify_admin_login(email_val, password_val):
            session["admin_logged_in"] = True
            session["admin_email"] = email_val
            next_url = request.args.get("next") or url_for("admin")
            return redirect(next_url)
        else:
            error = "Invalid credentials. Please verify your email and password."

    return render_template("login.html", error=error, email=email_val)


@app.route("/logout")
def logout():
    """Administrator Logout."""
    session.pop("admin_logged_in", None)
    session.pop("admin_email", None)
    return redirect(url_for("login"))


@app.route("/admin")
@admin_required
def admin():
    """Administrator Knowledge Management & Ingestion Dashboard."""
    return render_template("admin.html", admin_email=session.get("admin_email"))


# =====================================================================
# CHAT API
# =====================================================================
@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    """Process patient dental inquiry."""
    try:
        data = request.get_json() or {}
        message = data.get("message", "").strip()
        history = data.get("history", [])

        if not message:
            return jsonify({"success": False, "error": "Message cannot be empty."}), 400

        config = get_config()
        if not config.get("OPENROUTER_API_KEY"):
            return jsonify({
                "success": False, 
                "error": "OpenRouter API Key is missing. Please configure it in the Admin Settings panel."
            }), 400

        start_time = time.time()
        result = ask_dentist(query=message, chat_history=history)
        elapsed_sec = round(time.time() - start_time, 2)

        return jsonify({
            "success": True,
            "answer": result["answer"],
            "sources": result["sources"],
            "retrieved_count": result["retrieved_count"],
            "reranked_count": result["reranked_count"],
            "model_used": result["model_used"],
            "latency": f"{elapsed_sec}s"
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream_endpoint():
    """Process patient dental inquiry with real-time token streaming (ChatGPT-like)."""
    try:
        data = request.get_json() or {}
        message = data.get("message", "").strip()
        history = data.get("history", [])

        if not message:
            return jsonify({"success": False, "error": "Message cannot be empty."}), 400

        config = get_config()
        if not config.get("OPENROUTER_API_KEY"):
            return jsonify({
                "success": False, 
                "error": "OpenRouter API Key is missing. Please configure it in the Admin Settings panel."
            }), 400

        return Response(
            stream_dentist_response(query=message, chat_history=history),
            mimetype="text/event-stream"
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# =====================================================================
# ADMIN API - DOCUMENT MANAGEMENT
# =====================================================================
@app.route("/api/admin/documents", methods=["GET"])
@admin_required
def list_documents():
    """List dental books and documents currently stored in src/."""
    try:
        docs = []
        for file_path in SRC_DIR.glob("*.pdf"):
            size_mb = round(file_path.stat().st_size / (1024 * 1024), 2)
            docs.append({
                "filename": file_path.name,
                "size_mb": size_mb,
                "is_default": "orthodontic" in file_path.name.lower()
            })
        return jsonify({"success": True, "documents": docs})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/admin/upload", methods=["POST"])
@admin_required
def upload_document():
    """Upload a new dental textbook/PDF into src/."""
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded."}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "error": "Empty filename."}), 400

        if not file.filename.lower().endswith(".pdf"):
            return jsonify({"success": False, "error": "Only PDF files are supported."}), 400

        # Sanitize filename
        safe_filename = "".join(c for c in file.filename if c.isalnum() or c in "._- ")
        save_path = SRC_DIR / safe_filename
        file.save(str(save_path))

        return jsonify({
            "success": True,
            "filename": safe_filename,
            "message": f"Successfully uploaded {safe_filename}."
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# =====================================================================
# ADMIN API - PIPELINE EXECUTION
# =====================================================================
def run_pipeline_worker(pdf_path: str):
    """Background thread worker for pipeline execution."""
    global pipeline_state
    pipeline_state["is_running"] = True
    pipeline_state["percentage"] = 0
    pipeline_state["current_stage"] = "starting"
    pipeline_state["message"] = "Initializing dental RAG pipeline..."
    pipeline_state["logs"] = [f"[{time.strftime('%H:%M:%S')}] Started pipeline for {Path(pdf_path).name}"]
    pipeline_state["result"] = None
    pipeline_state["error"] = None

    def callback(stage: str, msg: str, pct: int, extra: dict):
        pipeline_state["current_stage"] = stage
        pipeline_state["message"] = msg
        pipeline_state["percentage"] = pct
        log_entry = f"[{time.strftime('%H:%M:%S')}] [{stage.upper()}] {msg}"
        pipeline_state["logs"].append(log_entry)

    try:
        result = run_full_ingestion_pipeline(pdf_path=pdf_path, progress_callback=callback)
        pipeline_state["result"] = result
        pipeline_state["is_running"] = False
        pipeline_state["current_stage"] = "complete"
        pipeline_state["percentage"] = 100
        pipeline_state["message"] = "Pipeline completed successfully!"
    except Exception as e:
        pipeline_state["is_running"] = False
        pipeline_state["current_stage"] = "error"
        pipeline_state["error"] = str(e)
        pipeline_state["message"] = f"Pipeline failed: {str(e)}"
        pipeline_state["logs"].append(f"[{time.strftime('%H:%M:%S')}] [ERROR] {str(e)}")


@app.route("/api/admin/run-pipeline", methods=["POST"])
@admin_required
def trigger_pipeline():
    """Trigger the 5-stage ingestion pipeline for a chosen document."""
    global pipeline_state
    if pipeline_state["is_running"]:
        return jsonify({"success": False, "error": "A pipeline task is already in progress."}), 400

    data = request.get_json() or {}
    filename = data.get("filename", "a_20guide_20to_20orthodontics.pdf")

    pdf_path = SRC_DIR / filename
    if not pdf_path.exists():
        # Fallback to any PDF in src/
        pdfs = list(SRC_DIR.glob("*.pdf"))
        if pdfs:
            pdf_path = pdfs[0]
        else:
            return jsonify({"success": False, "error": f"Document '{filename}' not found in src/."}), 404

    # Run in thread
    thread = threading.Thread(target=run_pipeline_worker, args=(str(pdf_path),), daemon=True)
    thread.start()

    return jsonify({"success": True, "message": f"Pipeline started for {pdf_path.name}."})


@app.route("/api/admin/pipeline-status", methods=["GET"])
@admin_required
def get_pipeline_status():
    """Poll pipeline execution state and live logs."""
    return jsonify(pipeline_state)


# =====================================================================
# ADMIN API - STATS & SETTINGS
# =====================================================================
@app.route("/api/admin/stats", methods=["GET"])
@admin_required
def get_stats():
    """Fetch live Pinecone and system status."""
    pinecone_info = get_pinecone_stats()
    config = get_config()

    # Mask API keys for security in UI display
    def mask_key(k: str) -> str:
        if not k or len(k) < 8:
            return "Not Configured"
        return f"{k[:4]}...{k[-4:]}"

    return jsonify({
        "success": True,
        "pinecone": pinecone_info,
        "config": {
            "openrouter_key": mask_key(config["OPENROUTER_API_KEY"]),
            "is_openrouter_set": bool(config["OPENROUTER_API_KEY"]),
            "pinecone_key": mask_key(config["PINECONE_API_KEY"]),
            "is_pinecone_set": bool(config["PINECONE_API_KEY"]),
            "index_name": config["PINECONE_INDEX_NAME"],
            "llm_model": config["LLM_MODEL"],
            "embedding_model": config["EMBEDDING_MODEL"]
        }
    })


@app.route("/api/admin/settings", methods=["POST"])
@admin_required
def update_settings():
    """Update API Keys and configurations in .env."""
    try:
        data = request.get_json() or {}

        # Save to .env file
        if not dotenv_path.exists():
            dotenv_path.touch()

        allowed_keys = [
            "OPENROUTER_API_KEY",
            "OPENROUTER_BASE_URL",
            "LLM_MODEL",
            "EMBEDDING_MODEL",
            "PINECONE_API_KEY",
            "PINECONE_INDEX_NAME",
            "PINECONE_CLOUD",
            "PINECONE_REGION"
        ]

        for k in allowed_keys:
            if k in data and data[k]:
                val = str(data[k]).strip()
                os.environ[k] = val
                set_key(str(dotenv_path), k, val)

        return jsonify({"success": True, "message": "Settings updated successfully."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  🦷 DENTAL AI CHATBOT & ORTHODONTIC KNOWLEDGE BASE")
    print("  - Patient Chat UI : http://localhost:5000/")
    print("  - Admin Dashboard  : http://localhost:5000/admin")
    print("="*60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
