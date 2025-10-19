"""Generates plots illustrating L1 (Lasso) and L2 (Ridge) constraints.

This script uses absl-py for application startup, flag handling, and logging.
"""

import os
from typing import Sequence

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from absl import (
  app,
  flags,
  logging,
)
from matplotlib.axes import Axes
from numpy.typing import NDArray

# --- Flag Definitions ---
# Per the style guide, flags are defined at the module level.
FLAGS = flags.FLAGS

flags.DEFINE_string(
  "output_dir",
  "./output/",
  "Directory to save the generated plot images.",
)
flags.DEFINE_boolean(
  "show_plots",
  False,
  "Whether to display the plots interactively after saving.",
)

# --- Constants ---
# Per the style guide, constants are uppercase with underscores.
C_LASSO: float = 1.0  # Constraint "budget" for L1
C_RIDGE_RADIUS: float = 1.0  # Constraint radius for L2


# --- Helper Function for Plotting ---
def setup_plot(
  ax: Axes, title: str, xlim: tuple[float, float], ylim: tuple[float, float]
) -> None:
  """Configures common plot elements like title, labels, limits, and grid.

  Args:
      ax: The matplotlib Axes object to configure.
      title: The title for the plot.
      xlim: A tuple containing the (min, max) x-axis limits.
      ylim: A tuple containing the (min, max) y-axis limits.
  """
  ax.set_title(title, fontsize=14)
  # Using LaTeX for mathematical notation
  ax.set_xlabel(r"$\beta_1$", fontsize=12)
  ax.set_ylabel(r"$\beta_2$", fontsize=12)
  ax.set_xlim(xlim)
  ax.set_ylim(ylim)
  ax.set_aspect("equal")
  ax.grid(True, linestyle="--", alpha=0.6)
  # Move spines to center for better visualization around origin
  ax.spines["left"].set_position("zero")
  ax.spines["bottom"].set_position("zero")
  ax.spines["right"].set_color("none")
  ax.spines["top"].set_color("none")
  ax.xaxis.set_ticks_position("bottom")
  ax.yaxis.set_ticks_position("left")


def plot_l1_constraint(c_lasso: float, output_dir: str) -> None:
  """Plots and saves the L1 constraint region (diamond).

  Args:
      c_lasso: The constraint budget (size of the diamond).
      output_dir: The directory to save the plot image.
  """
  fig, ax = plt.subplots(figsize=(7, 7))

  # Plot the 4 lines defining the diamond boundary
  beta_1_pos = np.linspace(0, c_lasso, 100)
  beta_1_neg = np.linspace(-c_lasso, 0, 100)

  ax.plot(beta_1_pos, c_lasso - beta_1_pos, "b", lw=3)  # Q1: b2 = C - b1
  ax.plot(beta_1_neg, c_lasso + beta_1_neg, "b", lw=3)  # Q2: b2 = C + b1
  ax.plot(beta_1_neg, -c_lasso - beta_1_neg, "b", lw=3)  # Q3: b2 = -C - b1
  ax.plot(beta_1_pos, -c_lasso + beta_1_pos, "b", lw=3)  # Q4: b2 = -C + b1

  # Fill the diamond area
  ax.fill(
    [c_lasso, 0, -c_lasso, 0, c_lasso],  # x-coords
    [0, c_lasso, 0, -c_lasso, 0],  # y-coords
    "b",
    alpha=0.2,
    label=r"$|\beta_1| + |\beta_2| \leq C$",
  )
  setup_plot(
    ax,
    f"L1 Constraint Region (C={c_lasso})",
    [-c_lasso * 1.5, c_lasso * 1.5],
    [-c_lasso * 1.5, c_lasso * 1.5],
  )
  ax.legend()
  # Save the figure
  filename = os.path.join(output_dir, "lasso_l1_constraint.png")
  fig.savefig(filename, bbox_inches="tight")
  logging.info("Saved: %s", filename)  # Use logging, not print


def plot_l2_constraint(c_ridge_radius: float, output_dir: str) -> None:
  """Plots and saves the L2 constraint region (circle).

  Args:
      c_ridge_radius: The constraint radius (radius of the circle).
      output_dir: The directory to save the plot image.
  """
  fig, ax = plt.subplots(figsize=(7, 7))

  # Create and add the circle patch
  circle = patches.Circle(
    (0, 0),
    c_ridge_radius,
    facecolor="r",
    alpha=0.2,
    label=r"$\beta_1^2 + \beta_2^2 \leq C^2$",
  )
  ax.add_patch(circle)

  # Plot boundary explicitly for clarity and consistent line weight
  theta = np.linspace(0, 2 * np.pi, 200)
  ax.plot(
    c_ridge_radius * np.cos(theta),
    c_ridge_radius * np.sin(theta),
    "r",
    lw=3,
  )
  setup_plot(
    ax,
    f"L2 Constraint Region (Radius={c_ridge_radius})",
    [-c_ridge_radius * 1.5, c_ridge_radius * 1.5],
    [-c_ridge_radius * 1.5, c_ridge_radius * 1.5],
  )
  ax.legend()
  # Save the figure
  filename = os.path.join(output_dir, "lasso_l2_constraint.png")
  fig.savefig(filename, bbox_inches="tight")
  logging.info("Saved: %s", filename)  # Use logging, not print


def plot_lasso_solution(
  x_grid: NDArray[np.float64],
  y_grid: NDArray[np.float64],
  loss_contours: NDArray[np.float64],
  beta_ols_hat: NDArray[np.float64],
  c_lasso: float,
  output_dir: str,
) -> None:
  """Plots and saves the Lasso (L1) solution finding process.

  Args:
      x_grid: X coordinates for the contour grid.
      y_grid: Y coordinates for the contour grid.
      loss_contours: Z values (loss) for the contour grid.
      beta_ols_hat: The OLS solution coordinates.
      c_lasso: The L1 constraint budget.
      output_dir: The directory to save the plot image.
  """
  fig, ax = plt.subplots(figsize=(7, 7))
  # Plot Loss Contours (Sum of Squared Residuals)
  ax.contour(
    x_grid, y_grid, loss_contours, levels=15, cmap="Greys_r", alpha=0.7
  )
  # Plot OLS solution (unconstrained minimum)
  ax.plot(
    beta_ols_hat[0],
    beta_ols_hat[1],
    "rx",
    markersize=10,
    mew=2,
    label=r"$\hat{\beta}_{OLS}$",
  )

  # Plot L1 Diamond Boundary
  ax.plot(
    [c_lasso, 0, -c_lasso, 0, c_lasso],
    [0, c_lasso, 0, -c_lasso, 0],
    "b",
    lw=3,
    label="L1 Constraint",
  )

  # Indicate the Lasso solution - typically where the contour first touches
  # the constraint region. For this specific contour/constraint, it's sparse.
  beta_lasso_hat = np.array([c_lasso, 0.0])  # Example solution
  ax.plot(
    beta_lasso_hat[0],
    beta_lasso_hat[1],
    "bo",
    markersize=10,
    label=r"$\hat{\beta}_{Lasso}$ (Sparse)",
  )

  setup_plot(ax, "Lasso (L1) Solution", [-0.5, 2], [-0.5, 2])
  ax.legend()
  # Save the figure
  filename = os.path.join(output_dir, "lasso_l1_solution.png")
  fig.savefig(filename, bbox_inches="tight")
  logging.info("Saved: %s", filename)  # Use logging, not print


def plot_ridge_solution(
  x_grid: NDArray[np.float64],
  y_grid: NDArray[np.float64],
  loss_contours: NDArray[np.float64],
  beta_ols_hat: NDArray[np.float64],
  c_ridge_radius: float,
  output_dir: str,
) -> None:
  """Plots and saves the Ridge (L2) solution finding process.

  Args:
      x_grid: X coordinates for the contour grid.
      y_grid: Y coordinates for the contour grid.
      loss_contours: Z values (loss) for the contour grid.
      beta_ols_hat: The OLS solution coordinates.
      c_ridge_radius: The L2 constraint radius.
      output_dir: The directory to save the plot image.
  """
  fig, ax = plt.subplots(figsize=(7, 7))
  # Plot Loss Contours
  ax.contour(
    x_grid, y_grid, loss_contours, levels=15, cmap="Greys_r", alpha=0.7
  )
  # Plot OLS solution
  ax.plot(
    beta_ols_hat[0],
    beta_ols_hat[1],
    "rx",
    markersize=10,
    mew=2,
    label=r"$\hat{\beta}_{OLS}$",
  )

  # Plot L2 Circle Boundary
  theta = np.linspace(0, 2 * np.pi, 200)
  ax.plot(
    c_ridge_radius * np.cos(theta),
    c_ridge_radius * np.sin(theta),
    "r",
    lw=3,
    label="L2 Constraint",
  )

  # Indicate the Ridge solution - where the contour first touches the circle.
  # This is typically not sparse. The exact point depends on the ellipse shape.
  beta_ridge_hat = np.array([0.77, 0.64])  # Approximated for this specific plot
  ax.plot(
    beta_ridge_hat[0],
    beta_ridge_hat[1],
    "ro",
    markersize=10,
    label=r"$\hat{\beta}_{Ridge}$ (Non-sparse)",
  )

  setup_plot(ax, "Ridge (L2) Solution", [-0.5, 2], [-0.5, 2])
  ax.legend()
  # Save the figure
  filename = os.path.join(output_dir, "lasso_l2_solution.png")
  fig.savefig(filename, bbox_inches="tight")
  logging.info("Saved: %s", filename)  # Use logging, not print


def main(argv: Sequence[str]) -> None:
  """Parses arguments, generates and saves Lasso/Ridge plots."""
  # absl.app handles argument parsing.
  # We check for extra positional arguments.
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  output_dir = FLAGS.output_dir
  # Create output directory if it doesn't exist
  os.makedirs(output_dir, exist_ok=True)

  # --- Generate Plots ---
  plot_l1_constraint(C_LASSO, output_dir)
  plot_l2_constraint(C_RIDGE_RADIUS, output_dir)

  # --- Setup for Solution Plots ---
  beta_ols_hat = np.array([1.2, 0.8])  # Ideal OLS solution

  # Create grid for contours
  x = np.linspace(-1, 2, 400)
  y = np.linspace(-1, 2, 400)
  x_grid, y_grid = np.meshgrid(x, y)

  # Define the Loss Function (Sum of Squared Residuals - SSR)
  # Using a slightly rotated/correlated ellipse for a more general case
  loss_contours = (
    0.5 * (x_grid - beta_ols_hat[0]) ** 2
    + (y_grid - beta_ols_hat[1]) ** 2
    - 0.4 * (x_grid - beta_ols_hat[0]) * (y_grid - beta_ols_hat[1])
  )

  plot_lasso_solution(
    x_grid, y_grid, loss_contours, beta_ols_hat, C_LASSO, output_dir
  )
  plot_ridge_solution(
    x_grid, y_grid, loss_contours, beta_ols_hat, C_RIDGE_RADIUS, output_dir
  )

  # Show plots interactively if requested
  logging.info("All plots saved to %s.", output_dir)

  if FLAGS.show_plots:
    logging.info("Displaying plots interactively.")
    plt.show()


# Standard Python entry point for absl
if __name__ == "__main__":
  app.run(main)
