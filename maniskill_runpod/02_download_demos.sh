#!/bin/bash
# =============================================================================
# STEP 2: DOWNLOAD DEMO TRAJECTORIES
# =============================================================================
# Downloads the raw demonstration data from ManiSkill.
#
# Usage: bash 02_download_demos.sh [TASK_NAME]
# Example: bash 02_download_demos.sh PickCube-v1
#          bash 02_download_demos.sh StackCube-v1
# =============================================================================

set -e

# --- Configuration ---
TASK="${1:-PickCube-v1}"    # Default task; override with first argument

echo "============================================================"
echo "  STEP 2: DOWNLOADING DEMOS FOR $TASK"
echo "============================================================"

# --- 2a. Download raw demo data ---
echo ""
echo "[1/2] Downloading demonstration data for $TASK..."
python -m mani_skill.utils.download_demo "$TASK"
echo "  ✓ Download complete"

# --- 2b. Inspect what was downloaded ---
echo ""
echo "[2/2] Inspecting downloaded data..."
DEMO_DIR="$HOME/.maniskill/demos/$TASK"

if [ ! -d "$DEMO_DIR" ]; then
    echo "  ✗ Demo directory not found at $DEMO_DIR"
    echo "    Check if the task name is correct."
    exit 1
fi

echo ""
echo "  Downloaded files:"
find "$DEMO_DIR" -type f -exec ls -lh {} \; | awk '{print "    " $NF " (" $5 ")"}'

# --- 2c. Quick inspection with Python ---
echo ""
python3 << 'PYEOF'
import h5py
import json
import os
import sys

task = os.environ.get("TASK", "PickCube-v1")
demo_dir = os.path.expanduser(f"~/.maniskill/demos/{task}")

# Find the trajectory file
h5_files = []
for root, dirs, files in os.walk(demo_dir):
    for f in files:
        if f.endswith('.h5'):
            h5_files.append(os.path.join(root, f))

if not h5_files:
    print("  ✗ No .h5 trajectory files found!")
    sys.exit(1)

for h5_path in h5_files:
    print(f"  Trajectory file: {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        traj_keys = sorted([k for k in f.keys() if k.startswith('traj_')])
        print(f"  Total trajectories: {len(traj_keys)}")
        
        if traj_keys:
            traj = f[traj_keys[0]]
            actions = traj['actions'][:]
            print(f"  First trajectory length: {actions.shape[0]} steps")
            print(f"  Action dimension: {actions.shape[1]}")
            
            # Check if visual data already exists
            has_visual = False
            if 'obs' in traj:
                def check_for_images(group, path=""):
                    for key in group.keys():
                        item = group[key]
                        if isinstance(item, h5py.Dataset):
                            if 'rgb' in key or 'depth' in key:
                                return True
                        elif isinstance(item, h5py.Group):
                            if check_for_images(item, f"{path}/{key}"):
                                return True
                    return False
                has_visual = check_for_images(traj['obs'])
            
            if has_visual:
                print("  Visual data: ✓ ALREADY HAS RGB/Depth")
                print("  → You may be able to skip Step 3 (replay)")
            else:
                print("  Visual data: ✗ No images (raw actions + states only)")
                print("  → Step 3 (replay) is REQUIRED to generate RGBD data")

    # Check for JSON metadata
    json_path = h5_path.replace('.h5', '.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as jf:
            meta = json.load(jf)
        env_kwargs = meta.get('env_kwargs', {})
        print(f"\n  Metadata from JSON:")
        print(f"    Control mode: {env_kwargs.get('control_mode', 'N/A')}")
        print(f"    Obs mode: {env_kwargs.get('obs_mode', 'N/A')}")
        print(f"    Sim backend: {meta.get('env_info', {}).get('env_kwargs', {}).get('sim_backend', 'N/A')}")

PYEOF

export TASK="$TASK"

echo ""
echo "============================================================"
echo "  DOWNLOAD COMPLETE!"
echo "============================================================"
echo ""
echo "  Task: $TASK"
echo "  Data: $DEMO_DIR"
echo ""
echo "  Next step: bash 03_replay_trajectories.sh $TASK"
echo "============================================================"
