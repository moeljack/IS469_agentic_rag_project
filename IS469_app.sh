#!/bin/bash
# Run the web chatbot (app.py) on a GPU node and reach it from your laptop through an SSH tunnel,
# the same way the cluster's Jupyter guide does. Submit from the project directory:
#     sbatch IS469_app.sh
# Then, once the job is running:
#     cat app_connect.<jobid>.txt          # the exact ssh command + URL for this job
# On your laptop (SMU network or VPN), paste the ssh command into a terminal / PowerShell, enter your cluster password,
# leave that window open, and open the URL in your browser. The app lives only as long as the job (--time below);
# stop it early with: scancel <jobid>
# Edit PROJECT_DIR / SCRATCH_DIR below (or export them) if your cluster paths differ.

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --time=00-02:01:00
#SBATCH --output=%u.%j.out

#SBATCH --partition=student
#SBATCH --account=is469
#SBATCH --qos=studentqos
#SBATCH --job-name=IS469_app

#################################################
##            ENVIRONMENT & CACHE SETUP        ##
#################################################

module purge
export PATH="$HOME/bin:$PATH"

PROJECT_DIR="${PROJECT_DIR:-/common/home/users/z/ziyi.lim.2022/IS469/IS469_agentic_rag_project}"
export SCRATCH_DIR="${SCRATCH_DIR:-/common/scratch/users/z/ziyi.lim.2022}"
LOGIN_HOST="${LOGIN_HOST:-origami.smu.edu.sg}"
export OLLAMA_MODELS="$SCRATCH_DIR/.ollama/models"
export HF_HOME="$SCRATCH_DIR/.cache/huggingface"
export TORCH_HOME="$SCRATCH_DIR/.cache/torch"
OLLAMA_PORT=$(( 11434 + (${SLURM_JOB_ID:-0} % 10000) ))     # 11434-21433, same scheme as the other job scripts
export OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT"

mkdir -p "$OLLAMA_MODELS" "$HF_HOME" "$TORCH_HOME"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

echo "Job $SLURM_JOB_ID on $(hostname) | git $(git rev-parse --short HEAD 2>/dev/null)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "WARNING: no GPU visible"

#################################################
##            PICK THE APP ADDRESS             ##
#################################################

# The tunnel goes laptop -> login node -> this node, so the app must listen on this node's cluster IP
# (127.0.0.1 would refuse the login node). A per-job port in 30000-39999 keeps clear of the Ollama ports and of
# other users' apps on the same node; if it is taken, try the next one.
NODE_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
[ -z "$NODE_IP" ] && NODE_IP=$(hostname -f)
APP_PORT=$(( 30000 + (${SLURM_JOB_ID:-0} % 10000) ))
port_free() { python3 -c "import socket,sys; s=socket.socket(); s.bind(('$NODE_IP', $1)); s.close()" 2>/dev/null; }
for _ in $(seq 1 50); do
    port_free "$APP_PORT" && break
    APP_PORT=$(( APP_PORT + 1 ))
done
export APP_HOST="$NODE_IP" APP_PORT

#################################################
##            START OLLAMA SERVICE             ##
#################################################

ollama serve > "ollama.${SLURM_JOB_ID}.log" 2>&1 &
OLLAMA_PID=$!
APP_PID=""
cleanup() {
    echo "Stopping app and Ollama..."
    [ -n "$APP_PID" ] && kill "$APP_PID" 2>/dev/null
    kill -9 "$OLLAMA_PID" 2>/dev/null
}
trap cleanup EXIT INT TERM

echo "Waiting for Ollama to initialize on $OLLAMA_HOST..."
if ! timeout 30 bash -c 'until curl -s "http://$OLLAMA_HOST/api/tags" > /dev/null; do sleep 1; done'; then
    echo "ERROR: Ollama server failed to start within 30 seconds."
    exit 1
fi

echo "Ollama is online. Pulling models..."
for MODEL in $(PYTHONPATH=. python3 -c "from src import config as c; print(' '.join({c.INTAKE_MODEL, c.PLANNER_MODEL, c.SUPERVISOR_MODEL, c.AUDITOR_MODEL}))"); do
    ollama pull "$MODEL"   # the tags set in src/config.py
done

#################################################
##            START THE WEB APP                ##
#################################################

# Same start-up as `python app.py` (init_app, then serve), but on this job's address instead of 0.0.0.0:8080.
# init_app() warms the models and reranker before the server listens, so the first page load can take a minute or two.
echo "Starting app on $APP_HOST:$APP_PORT ..."
python3 -c "
import os, app
app.init_app()
app.app.run(host=os.environ['APP_HOST'], port=int(os.environ['APP_PORT']), debug=False, threaded=True)
" &
APP_PID=$!

END_TIME=$(squeue -h -j "$SLURM_JOB_ID" -o %e 2>/dev/null)
CONNECT_FILE="$PROJECT_DIR/app_connect.${SLURM_JOB_ID}.txt"
cat > "$CONNECT_FILE" <<EOF
IS469 chatbot, job $SLURM_JOB_ID on $(hostname) ($NODE_IP), valid until ${END_TIME:-the job's time limit}.

1. On your laptop (SMU network or VPN), in a terminal / PowerShell window, run and keep it open:

   ssh -N -L ${APP_PORT}:${NODE_IP}:${APP_PORT} ${USER}@${LOGIN_HOST}

   (enter your cluster password; no further output is normal)

2. Open in your browser:

   http://127.0.0.1:${APP_PORT}

If port ${APP_PORT} is busy on your laptop, change only the first number, e.g. -L 18080:${NODE_IP}:${APP_PORT},
and open http://127.0.0.1:18080 instead.
Stop the app with: scancel $SLURM_JOB_ID
EOF
echo "================================================================"
cat "$CONNECT_FILE"
echo "================================================================"

# Report when the app is actually answering (models warm), so you know when to open the browser.
if timeout 900 bash -c "until curl -s http://$APP_HOST:$APP_PORT/status | grep -q '\"ready\": *true'; do
        kill -0 $APP_PID 2>/dev/null || exit 1; sleep 5; done"; then
    echo ">>> App is ready at http://127.0.0.1:${APP_PORT} (through the tunnel)."
else
    echo "WARNING: app not ready after 15 minutes or it exited; check the log above."
fi

wait "$APP_PID"
echo "App exited (status $?)."
