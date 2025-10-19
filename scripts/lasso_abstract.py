"""Generates plots illustrating 1D L1 (Lasso) and L2 (Ridge) penalties.

This script demonstrates why L1 regularization (Lasso) leads to sparse
solutions (coefficients at zero) while L2 (Ridge) does not.
It uses absl-py for application startup, flag handling, and logging.
"""

import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from absl import (
  app,
  flags,
  logging,  # Use absl.logging per style guide
)
from numpy.typing import NDArray

# --- Flag Definitions ---
# Per the style guide, flags are defined at the module level.
FLAGS = flags.FLAGS

flags.DEFINE_string(
  "output_dir",
  "./output/",
  "Directory to save the generated plot image.",
)
flags.DEFINE_boolean(
  "show_plots",
  False,
  "Whether to display the plot interactively after saving.",
)

# --- Constants ---
LAMBDA: float = 2.0  # Regularization strength "tax"

# --- Penalty Functions ---


def error_bowl(
  beta: NDArray[np.float64], beta_ideal: float
) -> NDArray[np.float64]:
  """Calculates the quadratic error (RSS) bowl.

  Args:
      beta: An array of beta coefficient values to evaluate.
      beta_ideal: The ideal, unpenalized beta value (bottom of the bowl).

  Returns:
      An array of loss values corresponding to the (beta - beta_ideal)^2.
  """
  # (beta - beta_ideal)^2 is a simple quadratic loss
  return (beta - beta_ideal) ** 2


def l1_penalty(beta: NDArray[np.float64], lam: float) -> NDArray[np.float64]:
  """Calculates the V-shaped L1 penalty.

  Args:
      beta: An array of beta coefficient values to evaluate.
      lam: The regularization strength (lambda).

  Returns:
      An array of penalty values corresponding to lam * |beta|.
  """
  return lam * np.abs(beta)


def l2_penalty(beta: NDArray[np.float64], lam: float) -> NDArray[np.float64]:
  """Calculates the U-shaped L2 penalty.

  Args:
      beta: An array of beta coefficient values to evaluate.
      lam: The regularization strength (lambda).

  Returns:
      An array of penalty values corresponding to lam * beta^2.
  """
  return lam * beta**2


def main(argv: Sequence[str]) -> None:
  """Generates and saves the 1D regularization plots."""
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  output_dir = FLAGS.output_dir
  os.makedirs(output_dir, exist_ok=True)

  # --- Setup data for plotting ---
  # We will test a range of possible beta values
  beta_range = np.linspace(-5, 5, 400)

  # --- Scenario 1: L1 (Lasso) with a "Weak" Feature ---
  # "ideal" beta is close to 0 (0.8).
  beta_ideal_weak = 0.8
  err_weak = error_bowl(beta_range, beta_ideal_weak)
  pen_l1 = l1_penalty(beta_range, LAMBDA)
  total_l1_weak = err_weak + pen_l1
  beta_min_l1_weak = beta_range[np.argmin(total_l1_weak)]

  # --- Scenario 2: L1 (Lasso) with a "Strong" Feature ---
  # "ideal" beta is far from 0 (3.0).
  beta_ideal_strong = 3.0
  err_strong = error_bowl(beta_range, beta_ideal_strong)
  # The penalty 'pen_l1' is the same as before
  total_l1_strong = err_strong + pen_l1
  beta_min_l1_strong = beta_range[np.argmin(total_l1_strong)]

  # --- Scenario 3: L2 (Ridge) with the "Weak" Feature ---
  # We use the same "weak" error bowl from Scenario 1 for comparison.
  pen_l2 = l2_penalty(beta_range, LAMBDA)
  total_l2_weak = err_weak + pen_l2
  beta_min_l2_weak = beta_range[np.argmin(total_l2_weak)]

  # --- Plotting ---
  # Note: 'seaborn-v0_8-whitegrid' is not part of the Google style guide,
  # but is a valid matplotlib style.
  plt.style.use("seaborn-v0_8-whitegrid")
  fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 18), sharex=True)
  fig.tight_layout(pad=5.0)  # Add padding for titles

  # Plot 1: L1 (Lasso) - Weak Feature -> Sparse Solution
  ax1.plot(beta_range, err_weak, "g--", label="Error (RSS) Bowl")
  ax1.plot(beta_range, pen_l1, "b--", label=r"L1 Penalty ($\lambda|\beta|$)")
  ax1.plot(
    beta_range,
    total_l1_weak,
    "r",
    label="Total Loss (Error + Penalty)",
    linewidth=3,
  )
  ax1.axvline(
    beta_min_l1_weak,
    color="k",
    linestyle=":",
    linewidth=2,
    label=f"Optimal $\\beta$ = {beta_min_l1_weak:.2f}",
  )
  ax1.set_title(
    (
      rf"Scenario 1: L1 (Lasso) - Weak Feature ($\lambda={LAMBDA}$)"
      + "\nMinimum is AT ZERO -> Sparsity"
    ),
    fontsize=16,
  )
  ax1.set_ylabel("Loss", fontsize=12)
  ax1.legend(fontsize=12)
  ax1.set_ylim(-1, 25)  # Set fixed y-axis for comparison

  # Plot 2: L1 (Lasso) - Strong Feature -> Shrunk Solution
  ax2.plot(beta_range, err_strong, "g--", label="Error (RSS) Bowl")
  ax2.plot(beta_range, pen_l1, "b--", label=r"L1 Penalty ($\lambda|\beta|$)")
  ax2.plot(
    beta_range,
    total_l1_strong,
    "r",
    label="Total Loss (Error + Penalty)",
    linewidth=3,
  )
  ax2.axvline(
    beta_ideal_strong,
    color="g",
    linestyle=":",
    linewidth=2,
    label=f"Original $\\beta$ = {beta_ideal_strong:.2f}",
  )
  ax2.axvline(
    beta_min_l1_strong,
    color="k",
    linestyle=":",
    linewidth=2,
    label=f"Optimal $\\beta$ = {beta_min_l1_strong:.2f}",
  )
  ax2.set_title(
    (
      rf"Scenario 2: L1 (Lasso) - Strong Feature ($\lambda={LAMBDA}$)"
      + "\nMinimum is SHRUNK, but not zero"
    ),
    fontsize=16,
  )
  ax2.set_ylabel("Loss", fontsize=12)
  ax2.legend(fontsize=12)
  ax2.set_ylim(-1, 25)

  # Plot 3: L2 (Ridge) - Weak Feature -> Shrunk Solution (No Sparsity)
  ax3.plot(beta_range, err_weak, "g--", label="Error (RSS) Bowl")
  ax3.plot(beta_range, pen_l2, "b--", label=r"L2 Penalty ($\lambda\beta^2$)")
  ax3.plot(
    beta_range,
    total_l2_weak,
    "r",
    label="Total Loss (Error + Penalty)",
    linewidth=3,
  )
  ax3.axvline(
    beta_ideal_weak,
    color="g",
    linestyle=":",
    linewidth=2,
    label=f"Original $\\beta$ = {beta_ideal_weak:.2f}",
  )
  ax3.axvline(
    beta_min_l2_weak,
    color="k",
    linestyle=":",
    linewidth=2,
    label=f"Optimal $\\beta$ = {beta_min_l2_weak:.2f}",
  )
  ax3.set_title(
    (
      rf"Scenario 3: L2 (Ridge) - Weak Feature ($\lambda={LAMBDA}$)"
      + "\nMinimum is SHRUNK, but NOT zero"
    ),
    fontsize=16,
  )
  ax3.set_xlabel(r"Coefficient Value ($\beta$)", fontsize=14)
  ax3.set_ylabel("Loss", fontsize=12)
  ax3.legend(fontsize=12)
  ax3.set_ylim(-1, 25)

  # Save the figure
  filename = os.path.join(output_dir, "lasso_1d_sparsity_visualization.png")
  plt.savefig(filename, bbox_inches="tight")

  # Show plot if flag is set
  logging.info("Plot saved to %s", filename)

  if FLAGS.show_plots:
    logging.info("Displaying plot...")
    plt.show()


if __name__ == "__main__":
  app.run(main)
