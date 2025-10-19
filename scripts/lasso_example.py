"""Generates plots fitting a linear (Lasso) model to a complex, oscillating function.

This script demonstrates model mismatch by showing how a simple linear model
fails to capture the underlying trend of y = x * sin(x), and how the fit
changes based on the *density* (number of points) of the training data.
"""

import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags, logging
from numpy.typing import NDArray
from sklearn import linear_model

# --- Flag Definitions ---
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
LASSO_ALPHA: float = 5  # Regularization strength
# Set x-range to 4*pi to show multiple oscillations
X_MAX: float = 4.0 * np.pi
# The number of data points (sparsity) for each of the 3 subplots
SUBPLOT_N_POINTS: tuple[int, int, int] = (10, 30, 100)


def complex_function(x: NDArray[np.float64]) -> NDArray[np.float64]:
  """The true, complex function we are trying to model.

  Args:
      x: An array of x-values.

  Returns:
      An array of y-values where y = x * sin(x).
  """
  return x * np.sin(x)


def main(argv: Sequence[str]) -> None:
  """Generates and saves the non-linear fit-by-sparsity plot."""
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  output_dir = FLAGS.output_dir
  os.makedirs(output_dir, exist_ok=True)

  # Create 3 horizontal subplots
  fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharey=True)
  fig.suptitle(
    r"Fitting a Linear (Lasso) Model to Complex ($y=x \cdot \sin(x)$) Data",
    fontsize=16,
    y=1.03,
  )

  # Create a dense x-axis for plotting the "true" smooth function
  x_dense = np.linspace(0, X_MAX, 300)
  y_dense_true = complex_function(x_dense)

  # Loop over each subplot axis and its corresponding number of points
  for ax, n_points in zip(axes, SUBPLOT_N_POINTS):
    # 1. Generate Data (No Noise)
    # Create x values from 0 to X_MAX with varying sparsity
    x_data = np.linspace(0, X_MAX, n_points)
    # Create the true y values
    y_data = complex_function(x_data)

    # 2. Fit Linear (Lasso) Model
    # sklearn requires X to be a 2D array [n_samples, n_features]
    X_train = x_data.reshape(-1, 1)

    model = linear_model.Lasso(alpha=LASSO_ALPHA)
    model.fit(X_train, y_data)
    y_pred = model.predict(X_train)

    # 3. Plot all components
    # Plot the "true" underlying function (smooth)
    ax.plot(
      x_dense,
      y_dense_true,
      "g--",
      label=r"True Function ($y = x \cdot \sin(x)$)",
      lw=2,
    )

    # Plot the discrete data points used for fitting
    ax.plot(
      x_data,
      y_data,
      "bo",  # 'bo' = blue circles
      label=f"Training Data (n={n_points})",
      markersize=8,
      alpha=0.7,
    )

    # Plot the fitted linear model (on the sparse points)
    # We need to predict on the dense x-axis to draw a continuous line
    y_pred_line = model.predict(x_dense.reshape(-1, 1))
    ax.plot(
      x_dense,
      y_pred_line,
      "r-",
      # Use raw f-string for LaTeX + variables
      label=rf"Lasso Fit ($y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}$)",
      lw=2,
    )

    # Plot the error (residuals) at each data point
    ax.vlines(
      x_data,
      y_pred,
      y_data,
      color="gray",
      linestyle=":",
      linewidth=1,
      label="Error (Residuals)",
    )

    # 4. Configure subplot
    ax.set_title(rf"Fit using {n_points} data points", fontsize=14)
    ax.set_xlabel(r"$x$", fontsize=12)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

  # Set the y-label only for the first plot (since they share a y-axis)
  axes[0].set_ylabel(r"$y$", fontsize=12)
  # Set a consistent y-limit
  axes[0].set_ylim(-X_MAX - 5, X_MAX + 5)

  plt.tight_layout(rect=[0, 0.03, 1, 0.95])

  # 5. Save and (optionally) show the plot
  filename = os.path.join(output_dir, "lasso_nonlinear_oscillating.png")
  plt.savefig(filename, bbox_inches="tight")

  if FLAGS.show_plots:
    logging.info("Displaying plot...")
    plt.show()
  else:
    logging.info("Plot saved to %s. Use --show_plots to display.", filename)


if __name__ == "__main__":
  app.run(main)
