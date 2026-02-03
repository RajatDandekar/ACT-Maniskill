#!/bin/bash
# =============================================================================
# STEP 1: REPLAY TRAJECTORIES WITH pd_ee_delta_pos (END-EFFECTOR CONTROL)
# =============================================================================
# Re-replays the raw demos using end-effector delta position control instead
# of joint delta position. EE control is much easier for imitation learning
# because the action space directly maps to gripper movement.
#
# Usage: bash 01_replay_ee_delta.sh
# =============================================================================

set -e

TASK="PickCube-v1"
CONTROL_MODE="pd_ee_delta_pos"
OBS_MODE="rgbd"

RAW_TRAJ="$HOME/.maniskill/demos/$TASK/motionplanning/trajectory.h5"
TRAJ_DIR=$(dirname "$RAW_TRAJ")
EXPECTED_OUTPUT="$TRAJ_DIR/trajectory.${OBS_MODE}.${CONTROL_MODE}.physx_cpu.h5"

echo "============================================================"
echo "  REPLAYING TRAJECTORIES WITH END-EFFECTOR CONTROL"
echo "============================================================"
echo ""
echo "  Task:         $TASK"
echo "  Control mode: $CONTROL_MODE  (was pd_joint_delta_pos)"
echo "  Obs mode:     $OBS_MODE"
echo "  Raw traj:     $RAW_TRAJ"
echo ""

# Check raw trajectory exists
if [ ! -f "$RAW_TRAJ" ]; then
    echo "  ✗ Raw trajectory not found: $RAW_TRAJ"
    echo "    Run: python -m mani_skill.utils.download_demo \"$TASK\""
    exit 1
fi
echo "  ✓ Raw trajectory found"

# Check if already replayed
if [ -f "$EXPECTED_OUTPUT" ]; then
    echo ""
    echo "  ⚠ Replayed file already exists: $EXPECTED_OUTPUT"
    read -p "  Overwrite? [y/N]: " -r
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "  Skipping replay. Using existing file."
        exit 0
    fi
fi

echo ""
echo "  Starting trajectory replay..."
echo "  (This may take several minutes)"
echo ""

# NOTE: Do NOT use --num-procs (doesn't exist in ManiSkill v3)
python -m mani_skill.trajectory.replay_trajectory \
    --traj-path "$RAW_TRAJ" \
    --use-first-env-state \
    -c "$CONTROL_MODE" \
    -o "$OBS_MODE" \
    --save-traj

echo ""

# Verify output
if [ ! -f "$EXPECTED_OUTPUT" ]; then
    echo "  ✗ Expected output not found: $EXPECTED_OUTPUT"
    echo "    Files in $TRAJ_DIR:"
    ls -la "$TRAJ_DIR"/*.h5
    exit 1
fi

# Quick verification
python3 << PYEOF
import h5py

path = "$EXPECTED_OUTPUT"

with h5py.File(path, 'r') as f:
    traj_keys = sorted([k for k in f.keys() if k.startswith('traj_')])
    print(f"  ✓ Trajectories: {len(traj_keys)}")
    
    traj = f[traj_keys[0]]
    
    def has_key(group, substr):
        for key in group.keys():
            if substr in key:
                return True
            if isinstance(group[key], h5py.Group):
                if has_key(group[key], substr):
                    return True
        return False
    
    has_rgb = has_key(traj, 'rgb')
    has_depth = has_key(traj, 'depth')
    print(f"  ✓ RGB data: {has_rgb}")
    print(f"  ✓ Depth data: {has_depth}")
    
    actions = traj['actions']
    print(f"  ✓ Action shape: {actions.shape} (should be smaller than joint control)")
PYEOF

echo ""
echo "============================================================"
echo "  REPLAY COMPLETE!"
echo "============================================================"
echo ""
echo "  Output: $EXPECTED_OUTPUT"
echo ""
echo "  Next: bash 02_train_ee_delta.sh"
echo "============================================================"
