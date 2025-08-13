import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

def categorical_colormap(categories, base_cmap='tab10'):
    """
    Creates a ListedColormap, BoundaryNorm, and category->code mapping for categorical data.
    """
    unique_categories = sorted(pd.Series(categories).unique(), key=lambda x: str(x))
    mapping = {cat: i for i, cat in enumerate(unique_categories)}
    base = plt.cm.get_cmap(base_cmap, len(unique_categories))
    cmap = ListedColormap(base.colors)
    norm = BoundaryNorm(range(len(unique_categories) + 1), cmap.N)
    return cmap, norm, mapping

def add_category_legend(ax, mapping, cmap, norm, title="Category"):
    """
    Adds a categorical legend to a matplotlib axis.
    """
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=cmap(norm(code)), markersize=10)
        for code in mapping.values()
    ]
    ax.legend(handles, mapping.keys(), title=title)

def bin_with_outliers(data, bin_edges, bin_labels):
    """
    Bins data into categories, adding <min and >max categories if needed.
    
    Parameters:
        data: array-like numeric values
        bin_edges: list of bin edges (ascending)
        bin_labels: list of labels for bins (must be len(bin_edges)-1)
    
    Returns:
        categories: pandas Series of category labels
        used_bins: list of bin edges actually used
        used_labels: list of labels actually used
    """
    used_bins = bin_edges.copy()
    used_labels = bin_labels.copy()
    
    min_val, max_val = np.min(data), np.max(data)
    
    # Handle lower outliers
    if min_val < bin_edges[0]:
        used_bins = [min_val - 1e-9] + used_bins
        used_labels = [f"<{bin_edges[0]}"] + used_labels
    
    # Handle upper outliers
    if max_val > bin_edges[-1]:
        used_bins = used_bins + [max_val + 1e-9]
        used_labels = used_labels + [f">{bin_edges[-1]}"]
    
    categories = pd.cut(data, bins=used_bins, labels=used_labels, include_lowest=True)
    return categories, used_bins, used_labels

# ---------------------------
# Test data for scatter
# ---------------------------
values = np.array([-2, 0.5, 2.8, 3.5, 7.9, 8.2, 12.5, 14.9, 18])

bin_edges = [0, 3, 8, 15]
bin_labels = ["Low (0–3)", "Medium (3–8)", "High (8–15)"]

categories, used_bins, used_labels = bin_with_outliers(values, bin_edges, bin_labels)
cmap, norm, mapping = categorical_colormap(categories)
codes = [mapping[cat] for cat in categories]

fig, ax = plt.subplots()
ax.scatter(range(len(values)), values, c=codes, cmap=cmap, norm=norm, s=100)
add_category_legend(ax, mapping, cmap, norm)
ax.set_title("Binned categories with outlier groups (scatter)")
ax.set_xlabel("Index")
ax.set_ylabel("Value")
plt.show()

# ---------------------------
# Test data for imshow
# ---------------------------
data_grid = np.array([
    [-5, 0.5, 2.8],
    [12.2, 3.3, 8.9],
    [14.0, 1.5, 20.2]
])

categories_grid, used_bins_g, used_labels_g = bin_with_outliers(
    data_grid.flatten(), bin_edges, bin_labels
)
cmap_grid, norm_grid, mapping_grid = categorical_colormap(categories_grid)
codes_grid = np.array([mapping_grid[cat] for cat in categories_grid]).reshape(data_grid.shape)

fig, ax = plt.subplots()
im = ax.imshow(codes_grid, cmap=cmap_grid, norm=norm_grid)
add_category_legend(ax, mapping_grid, cmap_grid, norm_grid)
ax.set_title("Binned categories with outlier groups (imshow)")
plt.show()