import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


def generate_default_colours(n):
    '''
    Generates a default colourmap in the case where the user does not specify 
    '''
    base_cmap = plt.get_cmap('tab10')
    return [base_cmap(i % base_cmap.N) for i in range(n)]


def get_categorical_colourmap(data, categories=None, colours=None, binary_cmap=False):
    """
    Returns a colourmap and normalization for categorical data.
    """

    if binary_cmap:
        # convert data to zero and non-zero
        data_indices = np.where(data == 0, 0, 1)
        categories = ['Yes', 'No']
        cmap = ListedColormap(['red', 'blue'])
        norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    else: 
        if categories is None:
            categories = np.unique(data)
        categories = list(categories)
        cat_to_index = {cat: idx for idx, cat in enumerate(categories)}
        data_indices = np.vectorize(cat_to_index.get)(data)
        print(data_indices)

        if colours is None or len(colours) < len(categories):
            colours = generate_default_colours(len(categories))

        cmap = ListedColormap(colours)
        norm = BoundaryNorm(np.arange(len(categories) + 1) - 0.5, cmap.N)

    return cmap, norm, categories, data_indices

def create_categorical_legend(categories, colours, ax=None, title="Legend", **kwargs):
    """
    Adds a legend with coloured patches for categorical data.

    Parameters:
    - categories: list of category labels
    - colours: list of colours (same length as categories)
    - ax: matplotlib axis (uses current if None)
    - title: title for the legend
    - kwargs: passed to ax.legend()
    """
    if ax is None:
        ax = plt.gca()

    handles = [Patch(color=colours[i], label=str(categories[i])) for i in range(len(categories))]
    ax.legend(handles=handles, title=title, **kwargs)


if __name__ == "__main__":

    fig, ax = plt.subplots(1, 3)
    fig.suptitle("Create Binary and Non-Binary Categorical Colourmap Using the Same Data")

    data = np.array([[0, 1, 0], [2, 0, 3]])
    
    # Non binary, no user-defined list of colours
    cmap, norm, categories, _ = get_categorical_colourmap(data)

    ax[0].imshow(data, cmap=cmap, norm=norm)
    ax[0].set_title("Non-binary")
    create_categorical_legend(categories, cmap.colors, ax=ax[0], loc='upper right')

    # Binary
    cmap, norm, categories, indices = get_categorical_colourmap(data, binary_cmap=True)

    ax[1].imshow(indices, cmap=cmap, norm=norm)
    ax[1].set_title("Binary")
    create_categorical_legend(categories, cmap.colors, ax=ax[1], loc='upper right')
    plt.show()

    
