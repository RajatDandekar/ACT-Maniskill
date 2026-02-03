#!/bin/bash
# =============================================================================
# STEP 4: TRAIN ACT POLICY (FIXED)
# =============================================================================
# Trains the ACT (Action Chunking with Transformers) policy using the
# replayed RGBD trajectories.
#
# Usage: bash 04_train_act.sh [TASK] [DEMO_PATH] [CONTROL_MODE]
# Example: bash 04_train_act.sh PickCube-v1 /path/to/replayed.h5 pd_joint_delta_pos
# =============================================================================

set -e

# --- Configuration ---
TASK="${1:-PickCube-v1}"
DEMO_PATH="${2}"
CONTROL_MODE="${3:-pd_joint_delta_pos}"

# Training hyperparameters (override via env vars)
TOTAL_ITERS="${TOTAL_ITERS:-30000}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-100}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-1}"
SEED="${SEED:-1}"
NUM_DEMOS="${NUM_DEMOS:-100}"
LOG_FREQ="${LOG_FREQ:-100}"
EVAL_FREQ="${EVAL_FREQ:-5000}"
SIM_BACKEND="${SIM_BACKEND:-physx_cpu}"

# Paths
MANISKILL_DIR="/workspace/ManiSkill"
ACT_DIR="$MANISKILL_DIR/examples/baselines/act"

echo "============================================================"
echo "  STEP 4: TRAINING ACT POLICY"
echo "============================================================"

# --- 4a. Verify prerequisites ---
echo ""
echo "[1/3] Checking prerequisites..."

# Check ACT code exists
if [ ! -f "$ACT_DIR/train_rgbd.py" ]; then
    echo "  ✗ ACT training script not found at $ACT_DIR/train_rgbd.py"
    echo "    Did you run 01_setup.sh?"
    exit 1
fi
echo "  ✓ ACT code found at $ACT_DIR"

# Auto-find demo path if not provided
if [ -z "$DEMO_PATH" ]; then
    echo "  No demo path provided, searching..."
    DEMO_DIR="$HOME/.maniskill/demos/$TASK"

    # Look for replayed RGBD files
    DEMO_PATH=$(find "$DEMO_DIR" -name "*.rgbd.*.h5" -type f 2>/dev/null | head -1)

    if [ -z "$DEMO_PATH" ]; then
        echo "  ✗ No replayed RGBD trajectory found for $TASK"
        echo "    Did you run 03_replay_trajectories.sh?"
        echo ""
        echo "    Files available:"
        find "$DEMO_DIR" -name "*.h5" -exec echo "      {}" \; 2>/dev/null
        exit 1
    fi
fi

if [ ! -f "$DEMO_PATH" ]; then
    echo "  ✗ Demo file not found: $DEMO_PATH"
    exit 1
fi
echo "  ✓ Demo path: $DEMO_PATH"

# Extract control mode from filename if possible
# Filename format: trajectory.rgbd.<control_mode>.physx_cpu.h5
BASENAME=$(basename "$DEMO_PATH")
EXTRACTED_CM=$(echo "$BASENAME" | sed -n 's/.*rgbd\.\(.*\)\.physx.*/\1/p')
if [ -n "$EXTRACTED_CM" ]; then
    CONTROL_MODE="$EXTRACTED_CM"
    echo "  ✓ Control mode (from filename): $CONTROL_MODE"
else
    echo "  ✓ Control mode (provided): $CONTROL_MODE"
fi

# Extract sim backend from filename
EXTRACTED_SB=$(echo "$BASENAME" | sed -n 's/.*\.\(physx_[a-z]*\)\.h5/\1/p')
if [ -n "$EXTRACTED_SB" ]; then
    SIM_BACKEND="$EXTRACTED_SB"
    echo "  ✓ Sim backend (from filename): $SIM_BACKEND"
fi

# --- 4b. Show training configuration ---
echo ""
echo "[2/3] Training configuration:"
echo "  Task:               $TASK"
echo "  Demo path:          $DEMO_PATH"
echo "  Control mode:       $CONTROL_MODE"
echo "  Sim backend:        $SIM_BACKEND"
echo "  Max episode steps:  $MAX_EPISODE_STEPS"
echo "  Total iterations:   $TOTAL_ITERS"
echo "  Num demos:          $NUM_DEMOS"
echo "  Eval envs:          $NUM_EVAL_ENVS"
echo "  Log frequency:      every $LOG_FREQ iters"
echo "  Eval frequency:     every $EVAL_FREQ iters"
echo "  Seed:               $SEED"
echo ""

# --- 4c. Build experiment name ---
EXP_NAME="act-${TASK}-rgbd-${NUM_DEMOS}_demos-seed${SEED}"

# --- 4d. Run training ---
echo "[3/3] Starting ACT training..."
echo "  Experiment: $EXP_NAME"
echo "  (Use tensorboard --logdir runs/ to monitor training)"
echo ""
echo "============================================================"
echo ""

# Change to the ACT directory so relative imports work
cd "$ACT_DIR"

python train_rgbd.py \
    --env-id "$TASK" \
    --demo-path "$DEMO_PATH" \
    --control-mode "$CONTROL_MODE" \
    --sim-backend "$SIM_BACKEND" \
    --max_episode_steps "$MAX_EPISODE_STEPS" \
    --total_iters "$TOTAL_ITERS" \
    --num_demos "$NUM_DEMOS" \
    --num-eval-envs "$NUM_EVAL_ENVS" \
    --log_freq "$LOG_FREQ" \
    --eval_freq "$EVAL_FREQ" \
    --seed "$SEED" \
    --exp-name "$EXP_NAME"

echo ""
echo "============================================================"
echo "  TRAINING COMPLETE!"
echo "============================================================"
echo ""
echo "  Checkpoints and logs saved in: $ACT_DIR/runs/$EXP_NAME"
echo ""
echo "  To monitor training:"
echo "    tensorboard --logdir $ACT_DIR/runs/ --port 6006"
echo ""
echo "  To train with W&B tracking, add --track:"
echo "    TOTAL_ITERS=50000 bash 04_train_act.sh $TASK $DEMO_PATH"
echo "============================================================"
