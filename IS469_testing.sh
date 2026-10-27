#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --time=00-02:01:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/common/home/users/z/ziyi.lim.2022/IS469/IS469_agentic_rag_project/%u.%j.out
#SBATCH --requeue

#SBATCH --partition=student
#SBATCH --account=is469
#SBATCH --qos=studentqos
#SBATCH --mail-user=yitong.ng.2022@smu.edu.sg
#SBATCH --job-name=IS469_testing

#################################################
##            ENVIRONMENT & CACHE SETUP        ##
#################################################

module purge

# Set project and scratch paths
PROJECT_DIR="/common/home/users/z/ziyi.lim.2022/IS469/IS469_agentic_rag_project"
export SCRATCH_DIR="/common/scratch/users/z/ziyi.lim.2022"
export OLLAMA_MODELS="$SCRATCH_DIR/.ollama/models"
export HF_HOME="$SCRATCH_DIR/.cache/huggingface"
export TORCH_HOME="$SCRATCH_DIR/.cache/torch"
export OLLAMA_HOST="127.0.0.1:11434"

mkdir -p "$OLLAMA_MODELS" "$HF_HOME" "$TORCH_HOME"

# Thread limits matching allocated CPUs
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

# Activate virtual environment
source "$PROJECT_DIR/.venv/bin/activate"

cd "$PROJECT_DIR"

#################################################
##            START OLLAMA SERVICE             ##
#################################################

# Launch Ollama in the background
ollama serve &
OLLAMA_PID=$!

# Ensure Ollama process terminates on exit, error, or cancellation
trap 'echo "Stopping Ollama (PID $OLLAMA_PID)..."; kill -9 $OLLAMA_PID 2>/dev/null' EXIT INT TERM

# Wait until Ollama API responds
echo "Waiting for Ollama to initialize on $OLLAMA_HOST..."
if ! timeout 30 bash -c 'until curl -s "http://$OLLAMA_HOST/api/tags" > /dev/null; do sleep 1; done'; then
    echo "ERROR: Ollama server failed to start within 30 seconds."
    exit 1
fi

echo "Ollama is online. Pulling model..."
ollama pull qwen3.5:4b

#################################################
##            EXECUTE YOUR RAG PROJECT         ##
#################################################

echo "Running RAG harness..."
python3 src/supervisor/harness.py --user tester --trace