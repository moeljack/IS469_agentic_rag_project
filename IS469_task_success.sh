#!/bin/bash
# Agent eval, task success (score_task_success.py) on the cluster GPU, with the local judge named in
# src/evaluation/agent_eval/configs/judge.yaml (provider ollama). Based on IS469_score.sh: same Slurm settings,
# caches and Ollama start-up, but it pulls agent_eval's judge and runs in the scoring environment (RAGAS).
#
# One-time setup on the login node, from the project directory (RAGAS pins its own LangChain versions, so it gets its
# own environment instead of the project's .venv):
#     python3 -m venv .venv
#     PIP_CACHE_DIR=/common/scratch/users/<x>/<you>/.cache/pip .venv/bin/pip install \
#         -r src/evaluation/agent_eval/requirements-scoring.txt
#
# Submit from the project directory, after the runs to score exist:
#     sbatch src/evaluation/agent_eval/IS469_task_success.sh
#     sbatch --export=ALL,TASK_ARGS="a1_agentic_qwen_robustness --suite robustness --workers 1" src/evaluation/agent_eval/IS469_task_success.sh
# TASK_ARGS are the arguments of score_task_success.py (default below). sbatch --export splits on commas, so keep
# commas out of TASK_ARGS. Judge scores are saved every --batch-size answers: if the job hits the time limit, submit the
# same command again and it continues. Afterwards, re-summarise on the login node with --no-judge (no GPU needed).
# Edit PROJECT_DIR / SCRATCH_DIR / SCORING_VENV below (or export them) if your cluster paths differ.

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --time=00-02:01:00
#SBATCH --output=%u.%j.out
#SBATCH --requeue

#SBATCH --partition=student
#SBATCH --account=is469
#SBATCH --qos=studentqos
#SBATCH --job-name=IS469_task_success

#################################################
##            ENVIRONMENT & CACHE SETUP        ##
#################################################

module purge
export PATH="$HOME/bin:$PATH"

PROJECT_DIR="${PROJECT_DIR:-/common/home/users/z/ziyi.lim.2022/IS469/IS469_agentic_rag_project}"
export SCRATCH_DIR="${SCRATCH_DIR:-/common/scratch/users/z/ziyi.lim.2022}"
SCORING_VENV="${SCORING_VENV:-$PROJECT_DIR/.venv}"
TASK_ARGS="${TASK_ARGS:-a1_agentic_qwen --by-type --workers 1}"
export OLLAMA_MODELS="$SCRATCH_DIR/.ollama/models"
export HF_HOME="$SCRATCH_DIR/.cache/huggingface"
export TORCH_HOME="$SCRATCH_DIR/.cache/torch"
PORT=$(( 11434 + (${SLURM_JOB_ID:-0} % 10000) ))
export OLLAMA_HOST="127.0.0.1:$PORT"

mkdir -p "$OLLAMA_MODELS" "$HF_HOME" "$TORCH_HOME"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

if [ ! -x "$SCORING_VENV/bin/python3" ]; then
    echo "ERROR: no scoring environment at $SCORING_VENV. Create it on the login node (see the top of this script)."
    exit 1
fi
source "$SCORING_VENV/bin/activate"
cd "$PROJECT_DIR"

# Fail in seconds, before Ollama starts, if the scoring packages are missing (an unfinished pip install, or the
# wrong environment).
if ! python3 -c "import ragas, langchain_huggingface, langchain_ollama, sentence_transformers, pandas, numpy, yaml" 2>import_error.$SLURM_JOB_ID.log; then
    echo "ERROR: $SCORING_VENV is missing scoring packages:"
    tail -n 1 "import_error.$SLURM_JOB_ID.log"
    echo "Install them on the login node: $SCORING_VENV/bin/pip install -r src/evaluation/agent_eval/requirements-scoring.txt"
    exit 1
fi
rm -f "import_error.$SLURM_JOB_ID.log"

echo "Job $SLURM_JOB_ID on $(hostname) | git $(git rev-parse --short HEAD 2>/dev/null)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "WARNING: no GPU visible"

# The judge must be a local Ollama model here: compute nodes may have no internet or API keys.
JUDGE=$(python3 -c "
import yaml
j = yaml.safe_load(open('src/evaluation/agent_eval/configs/judge.yaml'))
print(j['model'] if j['provider'] == 'ollama' else '')
")
if [ -z "$JUDGE" ]; then
    echo "ERROR: agent_eval/configs/judge.yaml is not an ollama judge. Set provider: ollama and model: <tag>, or score on your own machine."
    exit 1
fi
echo "Judge model: $JUDGE"

#################################################
##            START OLLAMA SERVICE             ##
#################################################

ollama serve > "ollama.${SLURM_JOB_ID}.log" 2>&1 &   # the server's own log goes to this file, so the job log shows only the scoring progress
OLLAMA_PID=$!
trap 'echo "Stopping Ollama (PID $OLLAMA_PID)..."; kill -9 $OLLAMA_PID 2>/dev/null' EXIT INT TERM

echo "Waiting for Ollama to initialize on $OLLAMA_HOST..."
if ! timeout 30 bash -c 'until curl -s "http://$OLLAMA_HOST/api/tags" > /dev/null; do sleep 1; done'; then
    echo "ERROR: Ollama server failed to start within 30 seconds."
    exit 1
fi

echo "Ollama is online. Pulling the judge..."
ollama pull "$JUDGE" || { echo "ERROR: could not pull $JUDGE"; exit 1; }

# Load the judge now, with the context size the scorer uses (Ollama reloads a model for a different num_ctx), and show
# where it runs: PROCESSOR should read "100% GPU". A CPU/GPU split means the model does not fit the card and every
# judge call is several times slower (the usual cause of RAGAS timeouts).
curl -s "http://$OLLAMA_HOST/api/generate" \
    -d "{\"model\": \"$JUDGE\", \"prompt\": \"\", \"keep_alive\": \"60m\", \"options\": {\"num_ctx\": 8192}}" > /dev/null
ollama ps

#################################################
##            SCORE TASK SUCCESS               ##
#################################################

echo "Running: python3 src/evaluation/agent_eval/score_task_success.py $TASK_ARGS"
python3 src/evaluation/agent_eval/score_task_success.py $TASK_ARGS
STATUS=$?
echo "Finished with status $STATUS. Scores are in src/evaluation/agent_eval/results/task_success/"
exit $STATUS