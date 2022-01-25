def get_subplots_dimensions(n_plots):
    """
    For given number of plots returns 'optimal' rows x columns distribution
    of subplots and figure size.
    Args:
        n_plots [int] - number of subplots to be includeed in the plot
    Returns:
        nrows [int] - suggested number of rows
        ncols [int] - suggested number of columns
        figsize [tuple[int, int]] - suggested figsize
    """
    if n_plots == 1:
        nrows = 1
        ncols = 1
        figsize = (12, 7)
    elif n_plots == 2:
        nrows = 1
        ncols = 2
        figsize = (13, 6)
    elif n_plots == 3:
        nrows = 1
        ncols = 3
        figsize = (20, 5)
    elif n_plots == 4:
        nrows = 2
        ncols = 2
        figsize = (14, 8)
    elif n_plots in [5, 6]:
        nrows = 2
        ncols = 3
        figsize = (20, 9)
    elif n_plots == 9:
        nrows = 3
        ncols = 3
        figsize = (18, 12)
    elif n_plots == 10:
        nrows = 2
        ncols = 5
        figsize = (20, 7)
    elif n_plots > 4:
        nrows = n_plots // 4 + 1
        ncols = 4
        figsize = (20, 7 + 5 * nrows)

    return nrows, ncols, figsize


# Review the following functions for usefulness
# def plot_for_all(func):
#     def wrapper(model_names, predictions, *args, **kwargs):
#         nrows, ncols, figsize = get_subplots_dimensions(len(models))
#         fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
#         for i, model_predictions in enumerate(predictions):
#             ax = fig.axes[i]
#             model_name = model_names[i]
#             func(model_name, model_predictions, ax=ax)
#         plt.tight_layout()
#         plt.show()

#     return wrapper
