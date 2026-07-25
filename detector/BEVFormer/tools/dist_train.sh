#!/usr/bin/env bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

export PYTHONPATH="$REPO_ROOT/mmcv:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    "$SCRIPT_DIR/train.py" "$CONFIG" --launcher pytorch ${@:3}
