#!/bin/bash
# Generate predictions for all models and score them.
#
# Usage:
#   ./run_all.sh                  # all models
#   ./run_all.sh --concurrency 20 # custom concurrency
#   ./run_all.sh --models claude-haiku-4.5 gpt-5-mini  # specific models

set -e

CONCURRENCY="${CONCURRENCY:-10}"
MODELS=""
EXTRA_ARGS=""

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        --models)
            shift
            while [[ $# -gt 0 && ! "$1" == --* ]]; do
                MODELS="$MODELS $1"
                shift
            done
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# If no models specified, get all from models.yaml
if [ -z "$MODELS" ]; then
    MODELS=$(python3 -c "
from config import MODEL_REGISTRY
for k in MODEL_REGISTRY:
    print(k)
")
fi

echo "Models: $(echo $MODELS | tr '\n' ' ')"
echo "Concurrency: $CONCURRENCY"
echo ""

FAILED=""
for model in $MODELS; do
    echo "=========================================="
    echo "  $model"
    echo "=========================================="
    if python3 generate.py --models "$model" --concurrency "$CONCURRENCY" $EXTRA_ARGS; then
        python3 run.py score --models "$model"
    else
        echo "  FAILED: $model"
        FAILED="$FAILED $model"
    fi
    echo ""
done

if [ -n "$FAILED" ]; then
    echo "Failed models:$FAILED"
    exit 1
else
    echo "All models completed successfully."
fi
