#!/bin/bash
# train-status.sh - Check training status on Mac Mini
#
# Usage: ./scripts/train-status.sh [session_name]

REMOTE_HOST="macmini"
SESSION="${1:-train}"
REMOTE_PATH="export PATH=/opt/homebrew/bin:\$PATH"

echo "=== Training Status: $SESSION ==="
echo ""

# Check if session exists
if ! ssh "$REMOTE_HOST" "$REMOTE_PATH && tmux has-session -t $SESSION 2>/dev/null"; then
    echo "Session '$SESSION' not found (training may have finished)"
    
    # Check for recent checkpoints
    echo ""
    echo "Recent checkpoints on Mac Mini:"
    ssh "$REMOTE_HOST" "ls -lt ~/projects/xtrans-demosaic/checkpoints/*.pt 2>/dev/null | head -5" || echo "  (none)"
    exit 0
fi

echo "Session '$SESSION' is active"
echo ""
echo "--- Last 30 lines of output ---"
ssh "$REMOTE_HOST" "$REMOTE_PATH && tmux capture-pane -t $SESSION -p | tail -30"
