#!/bin/bash
# RAG evaluation on the cluster (Stage 2 by default). Submit from the project directory:
#     sbatch IS469_eval.sh                                                        Stages 1 and 2
#     sbatch --export=ALL,EVAL_ARGS="--stages 2" IS469_eval.sh                    Stage 2 only
#     sbatch --export=ALL,EVAL_ARGS="--stages 1 2 3" IS469_eval.sh                 add the API generators (see below)
#     sbatch --export=ALL,EVAL_CMD="python3 <any command>" IS469_eval.sh          a trial slice of questions
# run_all.py is resumable: if the job hits the time limit, submit the same command again and it continues.
# Edit PROJECT_DIR / SCRATCH_DIR below (or export them) if your cluster paths differ from the harness script's.

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
#SBATCH --job-name=IS469_eval

#################################################
##            ENVIRONMENT & CACHE SETUP        ##
#################################################

module purge
export PATH="$HOME/bin:$PATH"

PROJECT_DIR="${PROJECT_DIR:-/common/home/users/z/ziyi.lim.2022/IS469/IS469_agentic_rag_project}"
export SCRATCH_DIR="${SCRATCH_DIR:-/common/scratch/users/z/ziyi.lim.2022}"
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

source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

echo "Job $SLURM_JOB_ID on $(hostname) | git $(git rev-parse --short HEAD 2>/dev/null)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "WARNING: no GPU visible"

#################################################
##            START OLLAMA SERVICE             ##
#################################################

ollama serve &
OLLAMA_PID=$!
trap 'echo "Stopping Ollama (PID $OLLAMA_PID)..."; kill -9 $OLLAMA_PID 2>/dev/null' EXIT INT TERM

echo "Waiting for Ollama to initialize on $OLLAMA_HOST..."
if ! timeout 30 bash -c 'until curl -s "http://$OLLAMA_HOST/api/tags" > /dev/null; do sleep 1; done'; then
    echo "ERROR: Ollama server failed to start within 30 seconds."
    exit 1
fi

echo "Ollama is online. Pulling model..."
ollama pull qwen3.5:4b

#################################################
##            RUN THE EVALUATION               ##
#################################################

# Scoring (Stage 4) is not run here: the judge needs an API key and internet, and compute nodes may not have either.
# Download results/ afterwards and run it on your own machine.
# EVAL_CMD overrides everything with one command line (for a trial slice of questions, see the header). Note that
# sbatch --export splits on commas, so keep commas out of EVAL_CMD.
EVAL_ARGS="${EVAL_ARGS:---stages 1 2 --skip-api}"
if [ -n "$EVAL_CMD" ]; then
    echo "Running: $EVAL_CMD"
    bash -c "$EVAL_CMD"
else
    echo "Running: python3 src/evaluation/rag_eval/run_all.py $EVAL_ARGS"
    python3 src/evaluation/rag_eval/run_all.py $EVAL_ARGS
fi
echo "Finished. Results are in src/evaluation/rag_eval/results/"
