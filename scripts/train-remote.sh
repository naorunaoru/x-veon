#!/bin/bash
# train-remote.sh - Sync code to Mac Mini and run training with completion notification
#
# Usage: ./scripts/train-remote.sh [training_script] [args...]
# Example: ./scripts/train-remote.sh train_v4.py --epochs 100 --lr 0.0001
#
# Environment:
#   NOTIFY=0         - Disable completion notification
#   SYNC=0           - Skip syncing code (use existing code on Mac Mini)
#   SESSION=train    - tmux session name (default: train)

set -e

# Configuration
REMOTE_HOST="macmini"
REMOTE_DIR="~/projects/xtrans-demosaic"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_ACTIVATE="source ~/ml/bin/activate"

# Spudnik webhook for notifications
WEBHOOK_URL="http://192.168.1.229:18789/hooks/wake"
WEBHOOK_TOKEN="35f35bf568b142bad7c8dafe1adeae0db3e6785d1e3fcdd7"

# Defaults
NOTIFY="${NOTIFY:-1}"
SYNC="${SYNC:-1}"
SESSION="${SESSION:-train}"

# Parse arguments
TRAIN_SCRIPT="${1:-train_v4.py}"
shift || true
TRAIN_ARGS="$@"

echo "=== X-Trans Remote Training ==="
echo "Script: $TRAIN_SCRIPT"
echo "Args: $TRAIN_ARGS"
echo "Session: $SESSION"
echo "Sync: $SYNC"
echo "Notify: $NOTIFY"
echo ""

# Step 1: Sync code to Mac Mini (if enabled)
if [ "$SYNC" = "1" ]; then
    echo ">>> Syncing code to $REMOTE_HOST..."
    # Use tar over ssh since rsync may not be available
    cd "$LOCAL_DIR"
    tar czf - \
        --exclude='checkpoints*' \
        --exclude='_obsolete' \
        --exclude='test_rafs' \
        --exclude='test_output' \
        --exclude='output' \
        --exclude='__pycache__' \
        --exclude='.git' \
        --exclude='*.pyc' \
        . | ssh "$REMOTE_HOST" "mkdir -p $REMOTE_DIR && cd $REMOTE_DIR && tar xzf -"
    echo ">>> Sync complete"
else
    echo ">>> Skipping sync (SYNC=0)"
fi

# Step 2: Build the training command with notification
NOTIFY_CMD=""
if [ "$NOTIFY" = "1" ]; then
    NOTIFY_CMD="&& curl -s -X POST '$WEBHOOK_URL' \
        -H 'Authorization: Bearer $WEBHOOK_TOKEN' \
        -H 'Content-Type: application/json' \
        -d '{\"text\":\"Training complete: $TRAIN_SCRIPT $TRAIN_ARGS\",\"mode\":\"now\"}'"
fi

FULL_CMD="cd $REMOTE_DIR && $VENV_ACTIVATE && python $TRAIN_SCRIPT $TRAIN_ARGS $NOTIFY_CMD"

# Mac Mini needs explicit PATH for homebrew
REMOTE_PATH="export PATH=/opt/homebrew/bin:\$PATH"

# Step 3: Check if session already exists
if ssh "$REMOTE_HOST" "$REMOTE_PATH && tmux has-session -t $SESSION 2>/dev/null"; then
    echo ">>> Session '$SESSION' already exists!"
    echo "    To view: ssh $REMOTE_HOST 'tmux attach -t $SESSION'"
    echo "    To kill: ssh $REMOTE_HOST 'tmux kill-session -t $SESSION'"
    exit 1
fi

# Step 4: Start training in detached tmux session
echo ">>> Starting training in tmux session '$SESSION'..."
ssh "$REMOTE_HOST" "$REMOTE_PATH && tmux new-session -d -s $SESSION '$FULL_CMD; exec bash'"

echo ""
echo "=== Training started ==="
echo "Monitor:  ssh $REMOTE_HOST 'tmux attach -t $SESSION'"
echo "Status:   ssh $REMOTE_HOST 'tmux capture-pane -t $SESSION -p | tail -30'"
echo "Kill:     ssh $REMOTE_HOST 'tmux kill-session -t $SESSION'"
echo ""
echo "You'll be notified via webhook when training completes."
