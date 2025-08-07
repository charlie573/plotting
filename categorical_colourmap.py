import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import cm
from matplotlib.patches import Patch

# -------------------------------------
# 🎨 Utility Functions
# -------------------------------------

def generate_default_colors(n):
    base_cmap = cm.get_cmap('tab10')
    return [base_cmap(i % base_cmap.N) for i in range(n)]

def get_categorical_colormap(data, categories=None, colors=None, binary_mode=False):
    """
    Returns a colormap and normalization for categorical data.
    """
    data = np.asarray(data)

    if binary_mode:
        data_indices = np.where(data == 0, 0, 1)
        categories = ['0', 'non-zero']
        cmap = ListedColormap(['red', 'blue'])
        norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    else:
        if categories is None:
            categories = np.unique(data)
        categories = list(categories)
        cat_to_index = {cat: idx for idx, cat in enumerate(categories)}
        data_indices = np.vectorize(cat_to_index.get)(data)

        if colors is None or len(colors) < len(categories):
            colors = generate_default_colors(len(categories))

        cmap = ListedColormap(colors)
        norm = BoundaryNorm(np.arange(len(categories) + 1) - 0.5, cmap.N)

    return cmap, norm, categories, data_indices

def create_categorical_legend(categories, colors, ax=None, title="Legend", **kwargs):
    """
    Adds a legend with colored patches for categorical data.

    Parameters:
    - categories: list of category labels
    - colors: list of colors (same length as categories)
    - ax: matplotlib axis (uses current if None)
    - title: title for the legend
    - kwargs: passed to ax.legend()
    """
    if ax is None:
        ax = plt.gca()

    handles = [Patch(color=colors[i], label=str(categories[i])) for i in range(len(categories))]
    ax.legend(handles=handles, title=title, **kwargs)

# -------------------------------------
# 🧪 EXAMPLES
# -------------------------------------

# 1️⃣ IM SHOW
print("Example 1: imshow (2D data with custom categories)")
data_imshow = np.array([[0, 1, 2], [2, 1, 0]])
cmap, norm, categories, _ = get_categorical_colormap(
    data_imshow,
    colors=['red', 'green', 'blue']
)

plt.imshow(data_imshow, cmap=cmap, norm=norm)
cbar = plt.colorbar(ticks=range(len(categories)))
cbar.ax.set_yticklabels(categories)
plt.title("imshow with categorical colormap")
plt.show()


# 2️⃣ SCATTER
print("Example 2: scatter (1D data with string labels)")
labels = np.array(["yes", "no", "maybe", "yes", "no", "maybe"])
x = np.arange(len(labels))
y = np.random.rand(len(labels))

cmap, norm, categories, indices = get_categorical_colormap(
    labels,
    colors=['green', 'red', 'orange']
)

fig, ax = plt.subplots()
sc = ax.scatter(x, y, c=indices, cmap=cmap, norm=norm)
ax.set_title("scatter with categorical labels")
ax.set_xticks(x)
ax.set_xticklabels(labels)
create_categorical_legend(categories, cmap.colors, ax=ax, loc='upper right')
plt.show()


# 3️⃣ BAR
print("Example 3: bar chart (with category-based colors)")
categories_bar = ['apple', 'banana', 'cherry']
values_bar = [5, 3, 7]

cmap, norm, categories, indices = get_categorical_colormap(
    categories_bar,
    colors=['red', 'yellow', 'purple']
)

colors_for_bars = [cmap(norm(i)) for i in indices]

fig, ax = plt.subplots()
ax.bar(categories_bar, values_bar, color=colors_for_bars)
ax.set_title("bar chart with category colors")
create_categorical_legend(categories, cmap.colors, ax=ax, loc='upper right')
plt.show()


# 4️⃣ BINARY MODE
print("Example 4: binary mode (0 vs non-zero)")
data_binary = np.array([[0, 1, 0], [2, 0, 3]])
cmap, norm, categories, indices = get_categorical_colormap(data_binary, binary_mode=True)

plt.imshow(indices, cmap=cmap, norm=norm)
cbar = plt.colorbar(ticks=range(len(categories)))
cbar.ax.set_yticklabels(categories)
plt.title("binary mode: 0 → red, non-zero → blue")
plt.show()
