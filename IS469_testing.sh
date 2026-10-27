#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --time=00-02:01:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/common/home/users/z/ziyi.lim.2022/IS469/%u.%j.out
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

# Route model downloads and temporary caches to your scratch storage
export SCRATCH_DIR="/common/scratch/users/z/ziyi.lim.2022"
export OLLAMA_MODELS="$SCRATCH_DIR/.ollama/models"
export HF_HOME="$SCRATCH_DIR/.cache/huggingface"
mkdir -p "$OLLAMA_MODELS" "$HF_HOME"

# Activate your virtual environment
# (Ensure this path points to the environment where your dependencies are installed)
source /common/home/users/z/ziyi.lim.2022/IS469/.venv/bin/activate

#################################################
##            START OLLAMA SERVICE             ##
#################################################

# Launch Ollama in the background on the allocated compute node
ollama serve &
OLLAMA_PID=$!

# Wait briefly for the Ollama daemon to come online
sleep 5

# Ensure your model is available on the node
ollama pull qwen3.5:4b

#################################################
##            EXECUTE YOUR RAG PROJECT         ##
#################################################

cd /common/home/users/z/ziyi.lim.2022/IS469/IS469_agentic_rag_project

# Replace main.py with the entry-point script of your project (e.g. supervisor.py or harness.py)
python3 src/supervisor/harness.py --user tester --trace

# Terminate the background Ollama service once the Python job finishes
kill $OLLAMA_PID
