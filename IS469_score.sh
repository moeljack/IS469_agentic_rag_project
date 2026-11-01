#!/bin/bash
# Stage 4 scoring on the cluster GPU, with the local judge named in configs/judge.yaml (provider ollama).
# Run this after the generation job (IS469_eval.sh) has finished, so the judge has the whole 11 GB card to itself.
# Submit from the project directory:
#     sbatch IS469_score.sh
# It runs score_retrieval, score_judge and score_grounding on every run in results/outputs. Each scorer resumes, so if
# the job hits the time limit, submit the same command again and it continues. analyze.py is not run here: copy
# results/ to your own machine afterwards and run it there (python src/evaluation/rag_eval/run_all.py --stages 4).
# Edit PROJECT_DIR / SCRATCH_DIR below (or export them) if your cluster paths differ.

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
#SBATCH --job-name=IS469_score

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

# The judge must be a local Ollama model here: compute nodes may have no internet or API keys.
JUDGE=$(python3 -c "
import yaml
j = yaml.safe_load(open('src/evaluation/rag_eval/rag_eval_approach/configs/judge.yaml'))
print(j['model'] if j['provider'] == 'ollama' else '')
")
if [ -z "$JUDGE" ]; then
    echo "ERROR: configs/judge.yaml is not an ollama judge. Set provider: ollama and model: <tag>, or score on your own machine."
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

#################################################
##            SCORE THE RUNS                   ##
#################################################

APPROACH=src/evaluation/rag_eval/rag_eval_approach
python3 $APPROACH/score_retrieval.py --all &&
python3 $APPROACH/score_judge.py --all &&
python3 $APPROACH/score_grounding.py --all
STATUS=$?
echo "Finished with status $STATUS. Scores are in src/evaluation/rag_eval/results/scores/"
exit $STATUS
