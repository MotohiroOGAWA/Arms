import numpy as np
import matplotlib.pyplot as plt

def compute_histogram(data, range_min, range_max, bin_width):
    """
    Compute a histogram of numerical data using specified bin range and width.

    This function calculates the frequency distribution of input values by grouping them 
    into bins defined by the given range and bin width.

    Parameters:
        data (list or np.ndarray): Input list of numerical values.
        range_min (float): Lower bound of the histogram range.
        range_max (float): Upper bound of the histogram range.
        bin_width (float): Width of each histogram bin.

    Returns:
        bin_edges (np.ndarray): The left edge of each bin (excluding the rightmost edge).
        counts (np.ndarray): Number of values falling into each corresponding bin.
    """
    # Create bin edges from range_min to range_max with specified bin_width
    bins = np.arange(range_min, range_max + bin_width, bin_width)

    # Compute the histogram
    counts, bin_edges = np.histogram(data, bins=bins)

    # Return the left edges of bins and corresponding counts
    return bin_edges[:-1], counts

def plot_histogram(bin_edges, counts, title='Histogram', xlabel='Value', ylabel='Frequency'):
    """
    Plot a histogram using precomputed bin edges and counts.

    Parameters:
        bin_edges (np.ndarray): The left edge of each bin.
        counts (np.ndarray): Number of elements in each bin.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    # Set the width of each bar as the difference between consecutive bin edges
    bin_width = bin_edges[1] - bin_edges[0]

    # Create the bar plot
    plt.bar(bin_edges, counts, width=bin_width, align='edge', edgecolor='black')

    # Set plot labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Display the plot
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_scatter(x, y, title='Scatter Plot', xlabel='X', ylabel='Y', xlim=None, ylim=None):
    """
    Plot a scatter plot using x and y values.

    Parameters:
        x (list or np.ndarray): Values for the x-axis.
        y (list or np.ndarray): Values for the y-axis.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        xlim (tuple or None): Limits for x-axis as (min, max).
        ylim (tuple or None): Limits for y-axis as (min, max).
    """
    plt.scatter(x, y, alpha=0.7, edgecolors='k')

    # Set plot labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Apply axis limits if provided
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    # Display grid and show plot
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_density_scatter(x, y, bins=100, cmap='viridis', title='Density Scatter Plot',
                         xlabel='X', ylabel='Y', xlim=None, ylim=None, colorbar=True,
                         ignore_nan=True, density_max=None):
    """
    Plot a 2D density scatter plot using histogram binning.

    Parameters:
        x (list or np.ndarray): Values for the x-axis.
        y (list or np.ndarray): Values for the y-axis.
        bins (int or [int, int]): Number of bins for the histogram in x and y.
        cmap (str): Colormap to use for density.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        xlim (tuple or None): Limits for x-axis as (min, max).
        ylim (tuple or None): Limits for y-axis as (min, max).
        colorbar (bool): Whether to show the colorbar.
        ignore_nan (bool): Whether to automatically remove NaN values. Default is True.
        density_max (float or None): Max value for color scale. Values above this use the same color.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if ignore_nan:
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]

    # Plot 2D histogram with optional vmax
    plt.hist2d(x, y, bins=bins, cmap=cmap, vmax=density_max)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    if colorbar:
        plt.colorbar(label='Density')

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()