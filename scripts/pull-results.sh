#!/bin/bash
# pull-results.sh - Pull checkpoints and results from Mac Mini
#
# Usage: ./scripts/pull-results.sh [what]
#   what: "checkpoints" | "latest" | "best" | "all"
#
# Examples:
#   ./scripts/pull-results.sh latest     # Just the latest checkpoint
#   ./scripts/pull-results.sh best       # Just best_model.pt
#   ./scripts/pull-results.sh checkpoints # All checkpoints
#   ./scripts/pull-results.sh all        # Everything including outputs

set -e

REMOTE_HOST="macmini"
REMOTE_DIR="~/projects/xtrans-demosaic"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

WHAT="${1:-latest}"

echo "=== Pulling results from Mac Mini ==="
echo "What: $WHAT"
echo ""

mkdir -p "$LOCAL_DIR/checkpoints"
mkdir -p "$LOCAL_DIR/output"

case "$WHAT" in
    latest)
        echo ">>> Pulling latest checkpoint..."
        LATEST=$(ssh "$REMOTE_HOST" "ls -t $REMOTE_DIR/checkpoints/*.pt 2>/dev/null | head -1")
        if [ -n "$LATEST" ]; then
            scp "$REMOTE_HOST:$LATEST" "$LOCAL_DIR/checkpoints/"
            echo "Pulled: $(basename $LATEST)"
        else
            echo "No checkpoints found"
        fi
        ;;
    
    best)
        echo ">>> Pulling best model..."
        scp "$REMOTE_HOST:$REMOTE_DIR/checkpoints/best_model.pt" "$LOCAL_DIR/checkpoints/" 2>/dev/null || echo "No best_model.pt found"
        ;;
    
    checkpoints)
        echo ">>> Pulling all checkpoints..."
        ssh "$REMOTE_HOST" "cd $REMOTE_DIR && tar czf - checkpoints/*.pt 2>/dev/null" | tar xzf - -C "$LOCAL_DIR" || echo "No checkpoints to pull"
        ls -lh "$LOCAL_DIR/checkpoints/"*.pt 2>/dev/null | tail -10
        ;;
    
    all)
        echo ">>> Pulling checkpoints..."
        ssh "$REMOTE_HOST" "cd $REMOTE_DIR && tar czf - checkpoints/ 2>/dev/null" | tar xzf - -C "$LOCAL_DIR" || true
        
        echo ">>> Pulling outputs..."
        ssh "$REMOTE_HOST" "cd $REMOTE_DIR && tar czf - output/ 2>/dev/null" | tar xzf - -C "$LOCAL_DIR" || true
        
        echo ">>> Pulling logs..."
        ssh "$REMOTE_HOST" "cd $REMOTE_DIR && tar czf - logs/ 2>/dev/null" | tar xzf - -C "$LOCAL_DIR" || true
        
        echo ""
        echo "Done. Local state:"
        du -sh "$LOCAL_DIR/checkpoints" "$LOCAL_DIR/output" "$LOCAL_DIR/logs" 2>/dev/null || true
        ;;
    
    *)
        echo "Unknown option: $WHAT"
        echo "Use: latest | best | checkpoints | all"
        exit 1
        ;;
esac

echo ""
echo "=== Done ==="
