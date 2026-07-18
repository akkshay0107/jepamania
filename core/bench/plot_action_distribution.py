import argparse
import sys
from pathlib import Path
from typing import cast

import h5py
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
import numpy as np
from core.actions import GAS_BRAKE_VALUES_NP, rescale_gas_np, to_continuous_action_np
from matplotlib.colors import LogNorm  # pyright: ignore[reportMissingImports]
from scipy.ndimage import gaussian_filter

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


def load_gamepad_actions(data_path: Path):
    """Loads all actions and telemetry inputs from HDF5 files in data_path.

    Returns:
        tuple: (actions, telemetry_actions) both as NumPy arrays,
               or (None, None) if no data.
    """
    if not data_path.exists():
        print(f"Data path does not exist: {data_path}")
        return None, None

    if data_path.is_file():
        h5_files = [data_path] if data_path.suffix == ".h5" else []
    else:
        h5_files = sorted(data_path.rglob("*.h5"))

    if not h5_files:
        print(f"No HDF5 files (*.h5) found in: {data_path}")
        return None, None

    print(f"Found {len(h5_files)} HDF5 files in {data_path}")

    all_actions = []
    all_telemetry_actions = []

    for file_path in h5_files:
        try:
            with h5py.File(file_path, "r") as f:
                if "actions" in f:
                    actions_ds = cast(h5py.Dataset, f["actions"])
                    actions = cast(np.ndarray, actions_ds[:])
                    if actions.ndim == 1 or (
                        actions.ndim == 2 and actions.shape[-1] != 3
                    ):
                        actions = to_continuous_action_np(actions)
                    actions = rescale_gas_np(actions.astype(np.float32))
                    all_actions.append(actions)
                else:
                    print(f"Warning: 'actions' dataset not found in {file_path.name}")

                telemetry = None
                if "observations/telemetry" in f:
                    ds = cast(h5py.Dataset, f["observations/telemetry"])
                    telemetry = cast(np.ndarray, ds[:])
                elif "observations" in f:
                    obs_grp = cast(h5py.Group, f["observations"])
                    if "telemetry" in obs_grp:
                        telemetry_ds = cast(h5py.Dataset, obs_grp["telemetry"])
                        telemetry = cast(np.ndarray, telemetry_ds[:])

                if telemetry is not None and telemetry.shape[1] >= 6:
                    telemetry_actions = telemetry[:, TELEM_ACT1_SLICE]
                    telemetry_actions = rescale_gas_np(telemetry_actions)
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
    """Prints summary statistics for gas, brake, and steering actions."""
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
    """Plots the 2D joint distribution of gas/brake and 1D histograms."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-Left: Gas vs Brake 2D Joint Distribution (Heatmap + Contours)
    ax_2d = axes[0, 0]
    counts, xedges, yedges = np.histogram2d(
        actions[:, 0], actions[:, 1], bins=45, range=[[0.0, 1.0], [0.0, 1.0]]
    )
    counts_t = counts.T
    X, Y = np.meshgrid(
        (xedges[:-1] + xedges[1:]) / 2.0, (yedges[:-1] + yedges[1:]) / 2.0
    )

    mesh = ax_2d.pcolormesh(
        xedges,
        yedges,
        np.maximum(counts_t, 1),
        cmap="inferno",
        norm=LogNorm(vmin=1.0, vmax=max(counts_t.max(), 10.0)),
    )
    fig.colorbar(mesh, ax=ax_2d, label="Frame Count (Log Scale)")

    smoothed = gaussian_filter(counts_t, sigma=1.0)
    ax_2d.contour(
        X,
        Y,
        smoothed,
        levels=7,
        colors="white",
        alpha=0.45,
        linewidths=0.9,
    )

    ax_2d.scatter(
        GAS_BRAKE_VALUES_NP[:, 0],
        GAS_BRAKE_VALUES_NP[:, 1],
        color="#00ffff",
        edgecolor="black",
        s=120,
        marker="*",
        label="Discrete Targets (6 pairs)",
        zorder=10,
    )
    ax_2d.set_title("Gas vs. Brake 2D Joint Distribution")
    ax_2d.set_xlabel("Gas Value [0.0, 1.0]")
    ax_2d.set_ylabel("Brake Value [0.0, 1.0]")
    ax_2d.legend(loc="upper left")
    ax_2d.grid(True, linestyle="--", alpha=0.4)

    # Top-Right: Steering Distribution
    ax_steer = axes[0, 1]
    ax_steer.hist(actions[:, 2], bins=50, color="#2cb0c5", edgecolor="black", alpha=0.7)
    ax_steer.set_title("Steering Distribution")
    ax_steer.set_xlabel("Steer Value [-1.0, 1.0]")
    ax_steer.set_ylabel("Frequency")
    ax_steer.grid(True, linestyle="--", alpha=0.6)

    # Bottom-Left: Gas Pedal Distribution
    ax_gas = axes[1, 0]
    ax_gas.hist(actions[:, 0], bins=35, color="#3bba5c", edgecolor="black", alpha=0.7)
    ax_gas.set_title("Gas Pedal 1D Distribution")
    ax_gas.set_xlabel("Gas Value [0.0, 1.0]")
    ax_gas.set_ylabel("Frequency")
    ax_gas.grid(True, linestyle="--", alpha=0.6)

    # Bottom-Right: Braking Distribution
    ax_brake = axes[1, 1]
    ax_brake.hist(actions[:, 1], bins=35, color="#f05454", edgecolor="black", alpha=0.7)
    ax_brake.set_title("Braking 1D Distribution")
    ax_brake.set_xlabel("Brake Value [0.0, 1.0]")
    ax_brake.set_ylabel("Frequency")
    ax_brake.grid(True, linestyle="--", alpha=0.6)

    plt.suptitle(
        "Trackmania Gamepad Input Distributions & Discretization Factorization",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"\nSaved distribution plot to: {output_path}")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze and plot gamepad action distributions from HDF5 datasets."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Path to data directory or dataset file (default: {DATA_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PLOT,
        help=f"Output path for saved plot (default: {OUTPUT_PLOT})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("Scanning for gamepad action datasets...")
    actions, telem_actions = load_gamepad_actions(args.data_dir)

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

    plot_distributions(actions, args.output)


if __name__ == "__main__":
    main()
