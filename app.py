import re
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify

# Ensure project root and src/supervisor are resolvable
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src" / "supervisor"))

from src.config import INTAKE_MODEL
from src.retrieval.user_store import UserStore
from supervisor import (
    MODEL, AUDITOR_MODEL, AUDIT_MIN_SCORE, DEFAULT_CANDIDATES,
    DEFAULT_K, DEFAULT_MAX_CHUNKS, DEFAULT_MAX_QUERIES,
    pin_ollama_keep_alive, warm_models
)
from src.supervisor.harness import build_supervisor

app = Flask(__name__)

DEFAULT_USER = "bingbling"
USER_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")

# State containers
sup_instance = None
store = None
conversations = {}  # In-memory per-user session message state: { user_id: [messages] }
init_status = {"ready": False, "step": "Initializing..."}


def init_app():
    """Pre-loads models and warms up the supervisor stack."""
    global sup_instance, store, init_status
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
        warm_models([INTAKE_MODEL, MODEL, AUDITOR_MODEL])

        init_status["step"] = "Loading reranker..."
        sup_instance.retrieval.warm()

        init_status["ready"] = True
        init_status["step"] = "Ready"
        print(">>> Startup complete. Supervisor ready for inputs.")
    except Exception as e:
        init_status["step"] = f"Failed to initialize: {str(e)}"
        print(f"Error during initialization: {e}")


def clean_user_id(raw):
    user_id = (raw or "").strip()
    return user_id if USER_ID_RE.match(user_id) else None


def profile_summary(profile):
    """What the frontend needs to show a user: id, company details and remembered facts."""
    return {
        "user_id": profile.get("user_id"),
        "created_at": profile.get("created_at"),
        "company": profile.get("company") or {},
        "facts": profile.get("facts") or {},
    }


@app.route("/")
def index():
    return render_template("index.html", default_user=DEFAULT_USER)


@app.route("/status", methods=["GET"])
def get_status():
    return jsonify(init_status)


@app.route("/users", methods=["GET"])
def get_users():
    """All users that have a profile in the user store."""
    if store is None:
        return jsonify({"error": "User store not ready"}), 503
    try:
        users = [profile_summary(p) for p in store.list_users()]
    except Exception as e:
        return jsonify({"error": f"Could not read user store: {e}"}), 500
    return jsonify({"users": users, "default": DEFAULT_USER})


@app.route("/users", methods=["POST"])
def create_user():
    """Create a profile for a new user (or return the existing one)."""
    if store is None:
        return jsonify({"error": "User store not ready"}), 503
    data = request.get_json() or {}
    user_id = clean_user_id(data.get("user"))
    if not user_id:
        return jsonify({"error": "Invalid user id (use letters, numbers, _ . -, max 64)"}), 400
    profile, created = store.ensure_profile(user_id)
    return jsonify({"user": profile_summary(profile), "created": created}), (201 if created else 200)


@app.route("/users/<user_id>", methods=["GET"])
def get_user(user_id):
    """One user's profile, re-read after each answer so learned facts show up."""
    if store is None:
        return jsonify({"error": "User store not ready"}), 503
    user_id = clean_user_id(user_id)
    if not user_id:
        return jsonify({"error": "Invalid user id"}), 400
    profile = store.get_profile(user_id)
    if profile is None:
        return jsonify({"error": f"No profile for {user_id}"}), 404
    return jsonify({"user": profile_summary(profile)})


@app.route("/reset", methods=["POST"])
def reset():
    """Clear the current conversation for a user, including any pending clarification.
    The profile and history in the user store are kept."""
    data = request.get_json() or {}
    user_id = clean_user_id(data.get("user"))
    if not user_id:
        return jsonify({"error": "Invalid user id"}), 400
    conversations.pop(user_id, None)
    if store is not None:
        store.pop_pending(user_id)
    return jsonify({"ok": True, "user": user_id})


@app.route("/chat", methods=["POST"])
def chat():
    if not init_status["ready"]:
        return jsonify({"error": "Application is still warming up. Please wait."}), 503

    data = request.get_json() or {}
    user_id = clean_user_id(data.get("user"))
    user_input = (data.get("message") or "").strip()

    if not user_id:
        return jsonify({"error": "Invalid user id (use letters, numbers, _ . -)"}), 400
    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    history = conversations.setdefault(user_id, [])

    # Append user turn
    history.append({"role": "user", "content": user_input})

    # Execute supervisor turn for this specific user
    try:
        out = sup_instance.handle_turn(user_id, history)
    except Exception as e:
        conversations.pop(user_id, None)
        return jsonify({"error": f"Supervisor error: {e}"}), 500

    # Store assistant response in history
    history.append({"role": "assistant", "content": out.message})

    # Clear temporary context if not asking for clarification (matching harness.py)
    if out.status != "clarify":
        conversations[user_id] = []

    return jsonify({
        "user": user_id,
        "reply": out.message,
        "status": out.status,
        "trace": getattr(out, "trace", []) or []
    })


if __name__ == "__main__":
    init_app()
    app.run(host="0.0.0.0", port=8080, debug=False)