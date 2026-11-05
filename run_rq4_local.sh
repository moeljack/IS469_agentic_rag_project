#!/bin/bash
# run_rq4_local.sh

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Loading API keys from .env..."
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
else
    echo "ERROR: .env file not found."
    exit 1
fi

echo "Checking if Ollama is running (needed for planning/auditing)..."
if ! curl -s "http://127.0.0.1:11434/api/tags" > /dev/null; then
    ollama serve &
    OLLAMA_PID=$!
    sleep 5
fi

echo "Running ONLY Stage 3 (API Generation) and Stage 4 (Scoring)..."
# The --stages 3 4 flag explicitly prevents Stages 1 and 2 from running.
python3 src/evaluation/rag_eval/run_all.py --stages 3 4

STATUS=$?

if [ -n "$OLLAMA_PID" ]; then
    echo "Stopping background Ollama service..."
    kill -9 $OLLAMA_PID 2>/dev/null
fi

echo "Finished with status $STATUS. Scores are in src/evaluation/rag_eval/results/scores/"
exit $STATUS