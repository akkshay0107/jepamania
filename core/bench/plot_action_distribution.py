import sys
from pathlib import Path
from typing import cast

import h5py
import matplotlib.pyplot as plt
import numpy as np

# Path-independent resolution based on file location
# to support execution from any directory
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = WORKSPACE_DIR / "win-client" / "data"
OUTPUT_PLOT = SCRIPT_DIR.parent / "out" / "action_distributions.png"

# Telemetry layout (TELEMETRY_FEATURES = 9):
#   [0]   speed
#   [1]   gear
#   [2]   rpm
#   [3:6] act1  — most recent previous action (gas, brake, steer)
#   [6:9] act2  — action before that
TELEM_ACT1_SLICE = slice(3, 6)


def load_gamepad_actions(data_dir: Path):
    """Loads all actions and telemetry inputs from HDF5 files in the data directory.

    Returns:
        tuple: (actions, telemetry_actions) both as NumPy arrays,
               or (None, None) if no data.
    """
    if not data_dir.exists():
        print(f"Data directory does not exist: {data_dir}")
        return None, None

    h5_files = list(data_dir.glob("*.h5"))
    if not h5_files:
        print(f"No HDF5 files (*.h5) found in data directory: {data_dir}")
        return None, None

    print(f"Found {len(h5_files)} HDF5 files in {data_dir}")

    all_actions = []
    all_telemetry_actions = []

    for file_path in h5_files:
        try:
            with h5py.File(file_path, "r") as f:
                if "actions" in f:
                    actions_ds = cast(h5py.Dataset, f["actions"])
                    actions = cast(np.ndarray, actions_ds[:])
                    all_actions.append(actions)
                else:
                    print(f"Warning: 'actions' dataset not found in {file_path.name}")

                if "observations" in f:
                    obs_grp = cast(h5py.Group, f["observations"])
                    if "telemetry" in obs_grp:
                        telemetry_ds = cast(h5py.Dataset, obs_grp["telemetry"])
                        telemetry = cast(np.ndarray, telemetry_ds[:])
                        # act1 (most recent previous action) is at indices 3:6
                        # within the 9-float telemetry vector.
                        if telemetry.shape[1] >= 6:
                            telemetry_actions = telemetry[:, TELEM_ACT1_SLICE]
                            all_telemetry_actions.append(telemetry_actions)
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")

    if not all_actions:
        return None, None

    actions_concat = np.concatenate(all_actions, axis=0)
    telem_concat = (
        np.concatenate(all_telemetry_actions, axis=0) if all_telemetry_actions else None
    )

    return actions_concat, telem_concat


def print_statistics(name: str, actions: np.ndarray):
    """Prints basic summary statistics for gas, brake, and steering actions."""
    print(f"\nSummary Statistics for {name} (Total Frames: {len(actions)}):")
    headers = ["Input Channel", "Min", "Max", "Mean", "Std Dev"]
    h0, h1, h2, h3, h4 = headers
    print(f"{h0:<15} | {h1:>8} | {h2:>8} | {h3:>8} | {h4:>8}")
    print("-" * 55)

    channels = ["Gas", "Braking", "Steering"]
    for i, channel_name in enumerate(channels):
        val = actions[:, i]
        print(
            f"{channel_name:<15} | {np.min(val):>8.4f} | {np.max(val):>8.4f} | "
            f"{np.mean(val):>8.4f} | {np.std(val):>8.4f}"
        )


def plot_distributions(actions: np.ndarray, output_path: Path):
    """Plots the histograms of gas, brake, and steering actions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)

    axes[0].hist(actions[:, 0], bins=30, color="#3bba5c", edgecolor="black", alpha=0.7)
    axes[0].set_title("Gas Pedal Distribution")
    axes[0].set_xlabel("Gas Value [0.0, 1.0]")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].hist(actions[:, 1], bins=30, color="#f05454", edgecolor="black", alpha=0.7)
    axes[1].set_title("Braking Distribution")
    axes[1].set_xlabel("Brake Value [0.0, 1.0]")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    axes[2].hist(actions[:, 2], bins=50, color="#2cb0c5", edgecolor="black", alpha=0.7)
    axes[2].set_title("Steering Distribution")
    axes[2].set_xlabel("Steer Value [-1.0, 1.0]")
    axes[2].grid(True, linestyle="--", alpha=0.6)

    plt.suptitle(
        "Trackmania Gamepad Input Distributions", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"\nSaved distribution plot to: {output_path}")
    plt.close()


def main():
    print("Scanning for gamepad action datasets...")
    actions, telem_actions = load_gamepad_actions(DATA_DIR)

    if actions is None:
        print(
            "\nNo gamepad action data found in the default directory. "
            "Exiting gracefully."
        )
        sys.exit(0)

    print_statistics("Recorded actions Dataset", actions)

    if telem_actions is not None:
        print_statistics(
            "Telemetry act1 features (Indices 3-5: gas, brake, steer)", telem_actions
        )

        diff = np.abs(actions - telem_actions)
        max_diff = np.max(diff)
        print(
            f"\nValidation: Maximum discrepancy between action and "
            f"telemetry inputs is {max_diff:.6f}"
        )
    else:
        print(
            "\nNote: Telemetry actions could not be validated "
            "(telemetry dataset not found or incomplete)."
        )

    plot_distributions(actions, OUTPUT_PLOT)


if __name__ == "__main__":
    main()
