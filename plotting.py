import matplotlib.pyplot as plt
import atexit

atexit.register(plt.show)

DEFAULT_COLORING = {'cmap':plt.cm.RdBu, 'vmin':-0.5, 'vmax':0.5}

def grid_plot(X, numpy_accessor, title, fig_titles=None, 
              coloring=DEFAULT_COLORING):

    if fig_titles == None:
        fig_titles = [""] * len(X)
    assert(len(X) == len(fig_titles))

    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(20,20))
    for x, ax, f_title in zip(X, axes.flat, fig_titles):
        im = ax.imshow(
            numpy_accessor(x), 
            cmap=coloring['cmap'], 
            vmin=coloring['vmin'], 
            vmax=coloring['vmax'],
        )
        ax.set_title(f_title, fontsize=8)
        ax.axis('off')
    fig.colorbar(im, ax=axes.ravel().tolist())
    fig.canvas.set_window_title(title)
    plt.show(block=False)
