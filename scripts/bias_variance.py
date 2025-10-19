import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from absl import app, flags, logging
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

FLAGS = flags.FLAGS

flags.DEFINE_string(
  "output_filename",
  "./output/bias_variance_graph.png",
  "Path to save the output graph image.",
)

flags.DEFINE_boolean(
  "show_plots",
  False,
  "Whether to display the plots interactively after saving.",
)

# Define parameters globally as constants [cite: 183]
_N_SAMPLES = 30
_DEGREES = [1, 4, 15]  # Model complexities
_TITLES = [
  "High Bias (Underfitting)",
  "Good Balance",
  "High Variance (Overfitting)",
]


def true_func(X):
  """The true underlying function we want to approximate."""
  return np.cos(1.5 * np.pi * X)


def main(argv):
  """Generates data, fits models, and plots the bias-variance tradeoff."""
  # Unused arguments are acceptable for `main` [cite: 34]
  del argv  # Unused.

  # Set a seed for reproducible results
  np.random.seed(0)

  # Generate random, noisy data based on the true function
  X = np.sort(np.random.rand(_N_SAMPLES))
  y = true_func(X) + np.random.randn(_N_SAMPLES) * 0.1

  # Set plot style
  sns.set_theme(style="whitegrid")
  plt.figure(figsize=(15, 5))

  # Loop through each model complexity
  for i, degree in enumerate(_DEGREES):
    ax = plt.subplot(1, len(_DEGREES), i + 1)

    # Create and fit the polynomial regression model
    # Use implicit line joining inside brackets [cite: 538]
    pipeline = Pipeline(
      [
        (
          "polynomial_features",
          PolynomialFeatures(degree=degree, include_bias=False),
        ),
        ("linear_regression", LinearRegression()),
      ]
    )
    pipeline.fit(X[:, np.newaxis], y)

    # Create a smooth set of points to plot the model's curve
    X_test = np.linspace(0, 1, 100)
    y_pred = pipeline.predict(X_test[:, np.newaxis])

    # Plot the results
    ax.plot(
      X_test,
      true_func(X_test),
      label="True Function",
      color="gray",
      linestyle="--",
    )
    ax.scatter(
      X, y, edgecolor="b", s=30, label="Data Samples", facecolors="none"
    )
    ax.plot(X_test, y_pred, label="Model Fit", color="red", linewidth=2)

    # Set plot labels and title
    ax.set_xlabel("X")
    if i == 0:
      ax.set_ylabel("y")
    ax.set_xlim((0, 1))
    ax.set_ylim((-2, 2))
    ax.legend(loc="upper right")
    ax.set_title(_TITLES[i])

  # Add a main title to the figure
  plt.suptitle("Bias-Variance Tradeoff Illustrated", fontsize=16)
  plt.tight_layout(
    rect=[0, 0, 1, 0.96]
  )  # Adjust layout to make room for suptitle

  # Ensure the output directory exists
  output_dir = os.path.dirname(FLAGS.output_filename)
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)

  # Save the figure instead of showing it
  plt.savefig(FLAGS.output_filename)
  logging.info("Graph saved to %s", FLAGS.output_filename)

  if FLAGS.show_plots:
    plt.show()
    logging.info("Displayed the plot interactively.")


# Standard main execution block for absl
if __name__ == "__main__":
  app.run(main)
