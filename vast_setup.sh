#!/bin/bash
# =============================================================================
# Vast.ai Setup & Training Script for GeoGuessr 1M DoRA Training
# =============================================================================
#
# Usage:
#   chmod +x vast_setup.sh
#   ./vast_setup.sh          # Interactive — walks through each step
#
# Prerequisites:
#   - Vast.ai account with credits
#   - Vast.ai API key (from https://vast.ai/console/account/)
# =============================================================================

set -e

# --- Config ---
PROJECT_DIR="/home/squishy33/claudeagent/GEOPROJECT"
DATA_DIR="/mnt/d/GEOPROJECT/data/osv5m_1m"
CHECKPOINT="$PROJECT_DIR/checkpoints/contrastive_best.pt"
REMOTE_WORK_DIR="/workspace/GEOPROJECT"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${GREEN}[STEP]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# =============================================================================
# Step 1: SSH Key
# =============================================================================
step_ssh_key() {
    log "Checking SSH key..."
    if [ -f ~/.ssh/id_rsa.pub ]; then
        log "SSH key already exists."
    else
        log "Generating SSH key..."
        mkdir -p ~/.ssh
        ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa
        eval "$(ssh-agent -s)"
        ssh-add ~/.ssh/id_rsa
    fi

    echo ""
    echo "========================================="
    echo "YOUR PUBLIC KEY (copy this to Vast.ai):"
    echo "========================================="
    cat ~/.ssh/id_rsa.pub
    echo "========================================="
    echo ""
    echo "Go to: https://vast.ai/console/account/ → SSH Keys → paste the above"
    read -p "Press Enter once you've added the key to Vast.ai..."
}

# =============================================================================
# Step 2: Install Vast CLI & Set API Key
# =============================================================================
step_install_cli() {
    log "Installing Vast.ai CLI..."
    pip install --upgrade vastai 2>/dev/null || pip install vastai

    read -p "Enter your Vast.ai API key: " API_KEY
    vastai set api-key "$API_KEY"
    log "CLI configured. Verifying..."
    vastai show user
}

# =============================================================================
# Step 3: Find & Rent GPU
# =============================================================================
step_rent_gpu() {
    log "Searching for RTX 5090 offers (on-demand, sorted by price)..."
    echo ""

    vastai search offers 'gpu_name=RTX_5090 num_gpus=1 disk_space>=200 inet_down>=100 reliability>0.95' \
        -o 'dph' --storage 200

    echo ""
    echo "Pick an offer from the list above."
    read -p "Enter the offer ID (first column): " OFFER_ID

    log "Renting offer $OFFER_ID with PyTorch image, 200GB disk, SSH access..."
    # pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel matches our local torch version
    # If this image fails to pull, Vast.ai will show an error — use 'pytorch/pytorch:latest' as fallback
    RESULT=$(vastai create instance "$OFFER_ID" \
        --image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
        --disk 200 \
        --ssh --direct \
        --onstart-cmd "echo 'ready'" \
        --raw)

    echo "$RESULT"

    # Extract instance ID from the response
    INSTANCE_ID=$(echo "$RESULT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('new_contract', d.get('instance_id','')))" 2>/dev/null || true)

    if [ -z "$INSTANCE_ID" ]; then
        read -p "Could not parse instance ID. Enter it manually: " INSTANCE_ID
    fi

    echo "$INSTANCE_ID" > /tmp/vast_instance_id
    log "Instance ID: $INSTANCE_ID"
    log "Waiting for instance to start..."

    for i in $(seq 1 60); do
        STATUS=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null | python3 -c "
import json,sys
d = json.load(sys.stdin)
status = d.get('actual_status','') or d.get('status_msg','')
print(status)
" 2>/dev/null || echo "unknown")

        if [ "$STATUS" = "running" ]; then
            log "Instance is running!"
            break
        fi
        echo "  Status: $STATUS ($i/60, checking every 10s)"
        sleep 10
    done

    # Get SSH connection info
    log "Getting SSH connection info..."
    vastai ssh-url "$INSTANCE_ID"
    SSH_URL=$(vastai ssh-url "$INSTANCE_ID" 2>/dev/null)
    echo "$SSH_URL" > /tmp/vast_ssh_url
    echo ""
    log "SSH command: $SSH_URL"
    echo ""
    log "Test connection with: $SSH_URL"
    read -p "Press Enter once you've verified SSH works..."
}

# =============================================================================
# Step 4: Upload Data
# =============================================================================
step_upload_data() {
    INSTANCE_ID=$(cat /tmp/vast_instance_id 2>/dev/null)

    if [ -z "$INSTANCE_ID" ]; then
        read -p "Enter instance ID: " INSTANCE_ID
    fi

    # Get SSH connection details via scp-url
    SCP_URL=$(vastai scp-url "$INSTANCE_ID" 2>/dev/null)
    SSH_URL=$(vastai ssh-url "$INSTANCE_ID" 2>/dev/null)

    # Parse host and port from ssh-url (format: ssh -p PORT root@HOST -L ...)
    SSH_PORT=$(echo "$SSH_URL" | grep -oP '(?<=-p )\d+')
    SSH_HOST=$(echo "$SSH_URL" | grep -oP 'root@[\w.-]+' | sed 's/root@//')

    if [ -z "$SSH_PORT" ] || [ -z "$SSH_HOST" ]; then
        echo "Could not parse SSH details from: $SSH_URL"
        read -p "Enter SSH host: " SSH_HOST
        read -p "Enter SSH port: " SSH_PORT
    fi

    log "SSH target: root@$SSH_HOST:$SSH_PORT"

    SSH_CMD="ssh -p $SSH_PORT -o StrictHostKeyChecking=no root@$SSH_HOST"
    RSYNC_BASE="rsync -avz --progress -e 'ssh -p $SSH_PORT -o StrictHostKeyChecking=no'"

    log "Creating remote directory structure..."
    $SSH_CMD "mkdir -p $REMOTE_WORK_DIR/{checkpoints,data/osv5m_1m/semantic_cells,geoguessr/data,geoguessr/model}"

    # Upload code
    log "Uploading code..."
    for f in __init__.py train.py eval_benchmark.py inference.py; do
        scp -P "$SSH_PORT" -o StrictHostKeyChecking=no \
            "$PROJECT_DIR/geoguessr/$f" "root@$SSH_HOST:$REMOTE_WORK_DIR/geoguessr/" 2>/dev/null && echo "  $f" || true
    done
    for f in __init__.py dataset.py geocells.py semantic_geocells.py download.py; do
        scp -P "$SSH_PORT" -o StrictHostKeyChecking=no \
            "$PROJECT_DIR/geoguessr/data/$f" "root@$SSH_HOST:$REMOTE_WORK_DIR/geoguessr/data/" 2>/dev/null && echo "  data/$f" || true
    done
    for f in __init__.py geolocator.py contrastive.py faiss_refinement.py; do
        scp -P "$SSH_PORT" -o StrictHostKeyChecking=no \
            "$PROJECT_DIR/geoguessr/model/$f" "root@$SSH_HOST:$REMOTE_WORK_DIR/geoguessr/model/" 2>/dev/null && echo "  model/$f" || true
    done

    # Create package init
    $SSH_CMD "touch $REMOTE_WORK_DIR/__init__.py"

    # Upload checkpoint (6.5MB)
    log "Uploading contrastive checkpoint (6.5MB)..."
    scp -P "$SSH_PORT" -o StrictHostKeyChecking=no \
        "$CHECKPOINT" "root@$SSH_HOST:$REMOTE_WORK_DIR/checkpoints/"

    # Upload semantic cells (~1.4GB)
    log "Uploading semantic cells (~1.4GB)..."
    eval $RSYNC_BASE "$DATA_DIR/semantic_cells/" "root@$SSH_HOST:$REMOTE_WORK_DIR/data/osv5m_1m/semantic_cells/"

    # Upload metadata CSVs (~248MB)
    log "Uploading metadata CSVs (~248MB)..."
    for f in metadata.csv metadata_train.csv metadata_val.csv; do
        scp -P "$SSH_PORT" -o StrictHostKeyChecking=no \
            "$DATA_DIR/$f" "root@$SSH_HOST:$REMOTE_WORK_DIR/data/osv5m_1m/" && echo "  $f"
    done

    # Upload images (~49GB — the big one)
    log "Uploading 1M images (~49GB) — this will take a while..."
    echo "  Estimated time at 200 Mbps: ~33 minutes"
    echo "  Estimated time at 100 Mbps: ~1.1 hours"
    echo "  Using rsync (resumable if interrupted — rerun this step to continue)"
    eval $RSYNC_BASE "$DATA_DIR/images/" "root@$SSH_HOST:$REMOTE_WORK_DIR/data/osv5m_1m/images/"

    log "Upload complete!"
    log "Verifying remote file count..."
    $SSH_CMD "ls $REMOTE_WORK_DIR/data/osv5m_1m/images/ | wc -l"
}

# =============================================================================
# Step 5: Install Dependencies & Launch Training
# =============================================================================
step_launch_training() {
    INSTANCE_ID=$(cat /tmp/vast_instance_id 2>/dev/null)
    SSH_URL=$(vastai ssh-url "$INSTANCE_ID" 2>/dev/null)
    SSH_PORT=$(echo "$SSH_URL" | grep -oP '(?<=-p )\d+')
    SSH_HOST=$(echo "$SSH_URL" | grep -oP 'root@[\w.-]+' | sed 's/root@//')

    if [ -z "$SSH_PORT" ] || [ -z "$SSH_HOST" ]; then
        read -p "Enter SSH host: " SSH_HOST
        read -p "Enter SSH port: " SSH_PORT
    fi

    SSH_CMD="ssh -p $SSH_PORT -o StrictHostKeyChecking=no root@$SSH_HOST"

    log "Installing dependencies on remote..."
    $SSH_CMD "pip install transformers peft open_clip_torch scikit-learn scipy"

    log "Writing training script to remote..."
    # Write script to file first — avoids fragile tmux quoting over SSH
    cat > /tmp/run_training.sh << 'TRAIN_SCRIPT'
#!/bin/bash
cd /workspace
export PYTHONPATH=/workspace

echo "=== Starting GeoGuessr 1M Training ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo ""

# batch_size=24 (safe for 32GB VRAM — 4x RTX4060's 8GB which ran bs=6)
# accumulation=3 → effective batch 72 (≈ original 60)
python -m GEOPROJECT.geoguessr.train \
    --epochs 15 \
    --batch-size 24 \
    --accumulation 3 \
    --data-dir GEOPROJECT/data/osv5m_1m \
    --geocell-dir GEOPROJECT/data/osv5m_1m/semantic_cells \
    --patience 3 \
    --contrastive-checkpoint GEOPROJECT/checkpoints/contrastive_best.pt \
    --checkpoint-dir GEOPROJECT/checkpoints \
    --num-workers 4 \
    2>&1 | tee /workspace/GEOPROJECT/training_1m.log

echo ""
echo "=== Training complete: $(date) ==="
TRAIN_SCRIPT

    scp -P "$SSH_PORT" -o StrictHostKeyChecking=no \
        /tmp/run_training.sh "root@$SSH_HOST:/workspace/run_training.sh"
    $SSH_CMD "chmod +x /workspace/run_training.sh"

    log "Launching training in tmux session 'train'..."
    $SSH_CMD "tmux new-session -d -s train '/workspace/run_training.sh'"

    echo ""
    echo "========================================="
    echo "  TRAINING IS RUNNING"
    echo "========================================="
    echo ""
    echo "Monitor progress:"
    echo "  $SSH_URL"
    echo "  tmux attach -t train"
    echo ""
    echo "When done, run: ./vast_setup.sh 6"
    echo "========================================="
}

# =============================================================================
# Step 6: Download Results
# =============================================================================
step_download_results() {
    INSTANCE_ID=$(cat /tmp/vast_instance_id 2>/dev/null)

    if [ -z "$INSTANCE_ID" ]; then
        read -p "Enter instance ID: " INSTANCE_ID
    fi

    SSH_URL=$(vastai ssh-url "$INSTANCE_ID" 2>/dev/null)
    SSH_PORT=$(echo "$SSH_URL" | grep -oP '(?<=-p )\d+')
    SSH_HOST=$(echo "$SSH_URL" | grep -oP 'root@[\w.-]+' | sed 's/root@//')

    if [ -z "$SSH_PORT" ] || [ -z "$SSH_HOST" ]; then
        read -p "Enter SSH host: " SSH_HOST
        read -p "Enter SSH port: " SSH_PORT
    fi

    log "Downloading best checkpoint..."
    scp -P "$SSH_PORT" -o StrictHostKeyChecking=no \
        "root@$SSH_HOST:$REMOTE_WORK_DIR/checkpoints/geolocator_best.pt" \
        "$PROJECT_DIR/checkpoints/geolocator_1m_best.pt"

    log "Downloading training log..."
    scp -P "$SSH_PORT" -o StrictHostKeyChecking=no \
        "root@$SSH_HOST:$REMOTE_WORK_DIR/training_1m.log" \
        "$PROJECT_DIR/checkpoints/training_1m.log"

    scp -P "$SSH_PORT" -o StrictHostKeyChecking=no \
        "root@$SSH_HOST:$REMOTE_WORK_DIR/checkpoints/training_log.json" \
        "$PROJECT_DIR/checkpoints/training_1m_log.json" 2>/dev/null || true

    log "Results downloaded to $PROJECT_DIR/checkpoints/"

    echo ""
    echo "========================================="
    echo "  NEXT: Destroy the instance to stop billing"
    echo "========================================="
    echo "  vastai destroy instance $INSTANCE_ID"
    echo ""
    echo "  Then evaluate locally:"
    echo "  source ~/geoenv/bin/activate"
    echo "  python -m GEOPROJECT.geoguessr.eval_benchmark --benchmark im2gps3k \\"
    echo "      --checkpoint GEOPROJECT/checkpoints/geolocator_1m_best.pt \\"
    echo "      --geocell-dir GEOPROJECT/data/osv5m_1m/semantic_cells"
    echo "========================================="
}

# =============================================================================
# Main Menu
# =============================================================================
echo ""
echo "========================================="
echo "  GeoGuessr 1M Training — Vast.ai Setup"
echo "========================================="
echo ""
echo "Steps:"
echo "  1) Generate SSH key & add to Vast.ai"
echo "  2) Install Vast CLI & set API key"
echo "  3) Find & rent RTX 5090 instance"
echo "  4) Upload data (~51GB total)"
echo "  5) Install deps & launch training"
echo "  6) Download results (after training)"
echo "  all) Run steps 1-5 sequentially"
echo ""
read -p "Which step? [1-6/all]: " STEP

case "$STEP" in
    1) step_ssh_key ;;
    2) step_install_cli ;;
    3) step_rent_gpu ;;
    4) step_upload_data ;;
    5) step_launch_training ;;
    6) step_download_results ;;
    all) step_ssh_key && step_install_cli && step_rent_gpu && step_upload_data && step_launch_training ;;
    *) err "Invalid step: $STEP" ;;
esac
