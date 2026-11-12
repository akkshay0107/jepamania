import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
import numpy as np
from core.actions import GAS_BRAKE_VALUES_NP, rescale_gas_np, to_continuous_action_np
from jaxtyping import Float
from matplotlib.colors import LogNorm  # pyright: ignore[reportMissingImports]
from scipy.ndimage import gaussian_filter

# Path-independent resolution based on file location
# to support execution from any directory
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = WORKSPACE_DIR / "win-client" / "data"
OUTPUT_PLOT = SCRIPT_DIR.parent / "out" / "action_distributions.png"

LATEST_CONTROL = slice(6, 9)


def load_gamepad_actions(
    data_path: Path,
) -> tuple[
    Float[np.ndarray, "actions 3"] | None,
    Float[np.ndarray, "paired_actions 3"] | None,
    float,
]:
    """Read executed controls, paired next-observation controls, and their max error."""
    if not data_path.exists():
        print(f"Data path does not exist: {data_path}")
        return None, None, 0.0

    if data_path.is_file():
        h5_files = [data_path] if data_path.suffix == ".h5" else []
    else:
        h5_files = sorted(data_path.rglob("*.h5"))

    if not h5_files:
        print(f"No HDF5 files (*.h5) found in: {data_path}")
        return None, None, 0.0

    print(f"Found {len(h5_files)} HDF5 files in {data_path}")

    all_actions = []
    all_telemetry_actions = []
    max_diff = 0.0

    for file_path in h5_files:
        try:
            with h5py.File(file_path, "r") as file:
                if "actions" not in file:
                    print(f"Warning: 'actions' dataset not found in {file_path.name}")
                    continue
                actions = np.asarray(file["actions"])
                if actions.ndim == 1 or (actions.ndim == 2 and actions.shape[1] == 1):
                    actions = to_continuous_action_np(actions.reshape(-1))
                if actions.ndim != 2 or actions.shape[1] != 3:
                    raise ValueError("Expected three control values per action.")

                ids = np.asarray(file.get("episode_id"))
                same_episode = (
                    ids[:-1] == ids[1:]
                    if ids.ndim == 1 and len(ids) == len(actions)
                    else None
                )
                final_observation_only = bool(
                    file.attrs.get("final_observation_only", False)
                )
                if final_observation_only:
                    if same_episode is None:
                        raise ValueError(
                            "Final observation rows require aligned episode IDs."
                        )
                    actions = actions[:-1][same_episode]
                if not len(actions):
                    continue
                actions = rescale_gas_np(actions.astype(np.float32))
                all_actions.append(actions)

                if (
                    same_episode is None
                    or not same_episode.any()
                    or "observations/telemetry" not in file
                ):
                    continue
                telemetry = np.asarray(file["observations/telemetry"], dtype=np.float32)
                if (
                    telemetry.ndim != 2
                    or len(telemetry) != len(ids)
                    or telemetry.shape[1] < 9
                ):
                    continue
                # The next observation carries the executed action in its newest slot.
                next_controls = rescale_gas_np(
                    telemetry[1:, LATEST_CONTROL][same_episode]
                )
                paired = (
                    actions if final_observation_only else actions[:-1][same_episode]
                )
                all_telemetry_actions.append(next_controls)
                max_diff = float(
                    np.maximum(max_diff, np.max(np.abs(paired - next_controls)))
                )
        except (OSError, ValueError, IndexError) as error:
            print(f"Error reading {file_path.name}: {error}")

    if not all_actions:
        return None, None, 0.0

    actions_concat = np.concatenate(all_actions, axis=0)
    telem_concat = (
        np.concatenate(all_telemetry_actions, axis=0) if all_telemetry_actions else None
    )

    return actions_concat, telem_concat, max_diff


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
    """Plot action distributions and save the resulting figure.

    Arguments:
      actions: Array of continuous [gas, brake, steer] actions.
      output_path: Destination path for the rendered plot.

    Returns:
      None. The figure is written to output_path.
    """
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
    actions, telem_actions, max_diff = load_gamepad_actions(args.data_dir)

    if actions is None:
        print(f"\nNo gamepad action data found in {args.data_dir}.")
        return

    print_statistics("Recorded actions", actions)

    if telem_actions is not None:
        print_statistics("Next-observation controls", telem_actions)
        print(
            f"\nMaximum discrepancy between recorded and next-observation controls "
            f"is {max_diff:.6f} across {len(telem_actions):,} paired transitions."
        )
    else:
        print(
            "\nNote: Telemetry actions could not be validated "
            "(telemetry dataset not found or incomplete)."
        )

    plot_distributions(actions, args.output)


if __name__ == "__main__":
    main()
