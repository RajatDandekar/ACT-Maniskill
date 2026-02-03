#!/bin/bash
# =============================================================================
# STEP 3: REPLAY TRAJECTORIES TO GENERATE RGBD DATA
# =============================================================================
# This is the CRITICAL step. Raw demos only have actions + states.
# We must REPLAY them to generate the RGBD observations needed for ACT.
#
# Uses ManiSkill's built-in replay tool which handles state initialization
# correctly (unlike manual replay which is error-prone).
#
# Usage: bash 03_replay_trajectories.sh [TASK_NAME] [CONTROL_MODE]
# Example: bash 03_replay_trajectories.sh PickCube-v1 pd_joint_delta_pos
# =============================================================================

set -e

# --- Configuration ---
TASK="${1:-PickCube-v1}"
CONTROL_MODE="${2:-pd_joint_delta_pos}"
OBS_MODE="rgbd"
NUM_PROCS="${3:-1}"                  # Increase if you have lots of CPU cores

echo "============================================================"
echo "  STEP 3: REPLAYING TRAJECTORIES WITH RGBD"
echo "============================================================"
echo ""
echo "  Task:         $TASK"
echo "  Control mode: $CONTROL_MODE"
echo "  Obs mode:     $OBS_MODE"
echo "  Num procs:    $NUM_PROCS"
echo ""

# --- 3a. Find the raw trajectory file ---
DEMO_DIR="$HOME/.maniskill/demos/$TASK"
RAW_TRAJ=""

# Look for the raw trajectory.h5 file
for subdir in motionplanning teleop rl; do
    candidate="$DEMO_DIR/$subdir/trajectory.h5"
    if [ -f "$candidate" ]; then
        RAW_TRAJ="$candidate"
        echo "  Found raw trajectory: $RAW_TRAJ"
        break
    fi
done

# Also check for trajectory directly in the task dir
if [ -z "$RAW_TRAJ" ] && [ -f "$DEMO_DIR/trajectory.h5" ]; then
    RAW_TRAJ="$DEMO_DIR/trajectory.h5"
    echo "  Found raw trajectory: $RAW_TRAJ"
fi

if [ -z "$RAW_TRAJ" ]; then
    echo "  ✗ No raw trajectory.h5 found in $DEMO_DIR"
    echo "    Available files:"
    find "$DEMO_DIR" -name "*.h5" -exec echo "      {}" \;
    echo ""
    echo "    If you already have a replayed .h5 file, you can skip to Step 4."
    exit 1
fi

# --- 3b. Determine the expected output filename ---
# ManiSkill replay tool names the output as:
#   trajectory.<obs_mode>.<control_mode>.<sim_backend>.h5
TRAJ_DIR=$(dirname "$RAW_TRAJ")
EXPECTED_OUTPUT="$TRAJ_DIR/trajectory.${OBS_MODE}.${CONTROL_MODE}.physx_cpu.h5"

# Check if already replayed
if [ -f "$EXPECTED_OUTPUT" ]; then
    echo ""
    echo "  ⚠ Replayed file already exists: $EXPECTED_OUTPUT"
    read -p "  Overwrite? [y/N]: " -r
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "  Skipping replay. Using existing file."
        echo ""
        echo "  Next step: bash 04_train_act.sh $TASK $EXPECTED_OUTPUT"
        exit 0
    fi
fi

# --- 3c. Run the replay ---
echo ""
echo "  Starting trajectory replay..."
echo "  (This may take a while depending on the number of trajectories)"
echo ""

python -m mani_skill.trajectory.replay_trajectory \
    --traj-path "$RAW_TRAJ" \
    --use-first-env-state \
    -c "$CONTROL_MODE" \
    -o "$OBS_MODE" \
    --save-traj \
    --num-procs "$NUM_PROCS"

echo ""

# --- 3d. Verify the output ---
# Find the generated file (name might vary slightly)
REPLAYED_FILE=""
for f in "$TRAJ_DIR"/*.${OBS_MODE}*.h5; do
    if [ -f "$f" ] && [ "$f" != "$RAW_TRAJ" ]; then
        REPLAYED_FILE="$f"
        break
    fi
done

if [ -z "$REPLAYED_FILE" ]; then
    echo "  ✗ Could not find replayed trajectory file!"
    echo "    Check the output above for errors."
    echo "    Files in $TRAJ_DIR:"
    ls -la "$TRAJ_DIR"
    exit 1
fi

echo "  ✓ Replayed trajectory saved to:"
echo "    $REPLAYED_FILE"
echo ""

# --- 3e. Verify the replayed data has RGBD ---
python3 << PYEOF
import h5py
import sys

replayed_path = "$REPLAYED_FILE"

print("  Verifying replayed data...")

with h5py.File(replayed_path, 'r') as f:
    traj_keys = sorted([k for k in f.keys() if k.startswith('traj_')])
    print(f"  Total trajectories: {len(traj_keys)}")
    
    if not traj_keys:
        print("  ✗ No trajectories found in replayed file!")
        sys.exit(1)
    
    traj = f[traj_keys[0]]
    
    # Check for observation data
    def find_datasets(group, path="", results=None):
        if results is None:
            results = []
        for key in group.keys():
            item = group[key]
            full_path = f"{path}/{key}"
            if isinstance(item, h5py.Dataset):
                results.append((full_path, item.shape, item.dtype))
            elif isinstance(item, h5py.Group):
                find_datasets(item, full_path, results)
        return results
    
    datasets = find_datasets(traj)
    
    has_rgb = any('rgb' in d[0] for d in datasets)
    has_depth = any('depth' in d[0] for d in datasets)
    has_actions = any('actions' in d[0] for d in datasets)
    
    print(f"  Actions:  {'✓' if has_actions else '✗'}")
    print(f"  RGB data: {'✓' if has_rgb else '✗'}")
    print(f"  Depth:    {'✓' if has_depth else '✗'}")
    
    if has_rgb and has_depth:
        # Show shapes
        for path, shape, dtype in datasets:
            if 'rgb' in path or 'depth' in path:
                print(f"    {path}: shape={shape}, dtype={dtype}")
                break
        print()
        print("  ✓ RGBD data is ready for ACT training!")
    else:
        print()
        print("  ✗ Missing visual data. Replay may have failed.")
        print("    All datasets found:")
        for path, shape, dtype in datasets[:20]:
            print(f"      {path}: {shape}")
        sys.exit(1)

PYEOF

echo ""
echo "============================================================"
echo "  REPLAY COMPLETE!"
echo "============================================================"
echo ""
echo "  Replayed file: $REPLAYED_FILE"
echo ""
echo "  Next step:"
echo "    bash 04_train_act.sh $TASK $REPLAYED_FILE"
echo "============================================================"
