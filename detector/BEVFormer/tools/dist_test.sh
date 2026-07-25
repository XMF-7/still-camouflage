#!/usr/bin/env bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29500}

export PYTHONPATH="$REPO_ROOT/mmcv:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    "$SCRIPT_DIR/test.py" "$CONFIG" "$CHECKPOINT" \
    --launcher pytorch ${@:4} --format-only --eval-options jsonfile_prefix=./test
