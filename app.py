import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify

# Ensure project root and src/supervisor are resolvable
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src" / "supervisor"))

from src.retrieval.user_store import UserStore
from supervisor import (
    MODEL, AUDITOR_MODEL, AUDIT_MIN_SCORE, DEFAULT_CANDIDATES, 
    DEFAULT_K, DEFAULT_MAX_CHUNKS, DEFAULT_MAX_QUERIES,
    pin_ollama_keep_alive, warm_models
)
from src.supervisor.harness import build_supervisor

app = Flask(__name__)

# State containers
sup_instance = None
conversations = {}  # In-memory per-user session message state: { user_id: [messages] }
init_status = {"ready": False, "step": "Initializing..."}

def init_app(user="bingbling"):
    """Pre-loads models and warms up the supervisor stack."""
    global sup_instance, init_status
    try:
        init_status["step"] = "Connecting store and building supervisor..."
        store = UserStore()
        
        sup_instance = build_supervisor(
            max_steps=4,
            store=store,
            k=DEFAULT_K,
            max_chunks=DEFAULT_MAX_CHUNKS,
            max_queries=DEFAULT_MAX_QUERIES,
            candidates=DEFAULT_CANDIDATES,
            audit_min_score=AUDIT_MIN_SCORE
        )
        
        init_status["step"] = "Pinning Ollama keep-alive..."
        pin_ollama_keep_alive()
        
        init_status["step"] = "Loading Ollama models (this may take a few moments)..."
        warm_models([os.environ.get("INTAKE_MODEL", "qwen3.5:4b"), MODEL, AUDITOR_MODEL])
        
        init_status["step"] = "Loading reranker..."
        sup_instance.retrieval.warm()
        
        init_status["ready"] = True
        init_status["step"] = "Ready"
        print(">>> Startup complete. Supervisor ready for inputs.")
    except Exception as e:
        init_status["step"] = f"Failed to initialize: {str(e)}"
        print(f"Error during initialization: {e}")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/status", methods=["GET"])
def get_status():
    return jsonify(init_status)

@app.route("/chat", methods=["POST"])
def chat():
    if not init_status["ready"]:
        return jsonify({"error": "Application is still warming up. Please wait."}), 503

    data = request.get_json() or {}
    user_id = data.get("user", "bingbling").strip()
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    if user_id not in conversations:
        conversations[user_id] = []

    # Append user turn
    conversations[user_id].append({"role": "user", "content": user_input})

    # Execute supervisor turn
    out = sup_instance.handle_turn(user_id, conversations[user_id])

    # Store assistant response in history
    conversations[user_id].append({"role": "assistant", "content": out.message})

    # Clear temporary context if not asking for clarification (matching harness.py)
    if out.status != "clarify":
        conversations[user_id] = []

    return jsonify({
        "reply": out.message,
        "status": out.status,
        "trace": out.trace if hasattr(out, "trace") else []
    })

if __name__ == "__main__":
    init_app(user="bingbling")
    app.run(host="0.0.0.0", port=8080, debug=False)