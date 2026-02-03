#!/bin/bash
# =============================================================================
# STEP 2: TRAIN ACT WITH pd_ee_delta_pos + ALL DEMOS
# =============================================================================
# Key changes from the previous run:
#   1. pd_ee_delta_pos control (easier action space for imitation learning)
#   2. All available demos (not just 100)
#   3. Correct argument format (underscores for some, hyphens for others)
#   4. --sim-backend physx_cpu and --max_episode_steps 100 included
#
# Usage: bash 02_train_ee_delta.sh
# =============================================================================

set -e

TASK="PickCube-v1"
CONTROL_MODE="pd_ee_delta_pos"
DEMO_PATH="$HOME/.maniskill/demos/$TASK/motionplanning/trajectory.rgbd.${CONTROL_MODE}.physx_cpu.h5"

# Training hyperparameters
TOTAL_ITERS="${TOTAL_ITERS:-30000}"
NUM_DEMOS="${NUM_DEMOS:-0}"            # 0 = use ALL available demos
SEED="${SEED:-1}"
EXP_NAME="act-PickCube-v1-rgbd-ee_delta-all_demos-seed${SEED}"

# Paths
ACT_DIR="/workspace/ManiSkill/examples/baselines/act"

echo "============================================================"
echo "  TRAINING ACT WITH END-EFFECTOR CONTROL"
echo "============================================================"
echo ""
echo "  Task:             $TASK"
echo "  Control mode:     $CONTROL_MODE"
echo "  Demo path:        $DEMO_PATH"
echo "  Num demos:        ${NUM_DEMOS} (0 = all)"
echo "  Total iterations: $TOTAL_ITERS"
echo "  Experiment:       $EXP_NAME"
echo ""

# Check demo file exists
if [ ! -f "$DEMO_PATH" ]; then
    echo "  ✗ Demo file not found: $DEMO_PATH"
    echo "    Did you run 01_replay_ee_delta.sh first?"
    exit 1
fi
echo "  ✓ Demo file found"

# Check ACT code exists
if [ ! -f "$ACT_DIR/train_rgbd.py" ]; then
    echo "  ✗ ACT code not found at $ACT_DIR/train_rgbd.py"
    exit 1
fi
echo "  ✓ ACT code found"

echo ""
echo "  Starting training..."
echo "  Monitor with: tensorboard --logdir $ACT_DIR/runs/ --port 6006 --bind_all"
echo ""
echo "============================================================"
echo ""

cd "$ACT_DIR"

# IMPORTANT: Argument format is a mix of hyphens and underscores.
# Hyphens:     --env-id, --demo-path, --control-mode, --sim-backend, --num-eval-envs, --exp-name
# Underscores: --max_episode_steps, --total_iters, --num_demos, --log_freq, --eval_freq

python train_rgbd.py \
    --env-id "$TASK" \
    --demo-path "$DEMO_PATH" \
    --control-mode "$CONTROL_MODE" \
    --sim-backend physx_cpu \
    --max_episode_steps 100 \
    --total_iters "$TOTAL_ITERS" \
    --num_demos "$NUM_DEMOS" \
    --num-eval-envs 1 \
    --log_freq 100 \
    --eval_freq 5000 \
    --seed "$SEED" \
    --exp-name "$EXP_NAME"

echo ""
echo "============================================================"
echo "  TRAINING COMPLETE!"
echo "============================================================"
echo ""
echo "  Checkpoints saved in: $ACT_DIR/runs/$EXP_NAME/"
echo ""
echo "  To evaluate the best checkpoint:"
echo "    bash 03_evaluate_ee_delta.sh"
echo "============================================================"
