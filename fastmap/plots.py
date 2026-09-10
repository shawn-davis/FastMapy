"""Optional 2D/3D plotting and visual reduction helpers."""

import numpy as np


def reduce_for_plot(embedding, method="tsne", n_components=2, random_state=None, **kwargs):
    """Reduce an existing embedding to two or three dimensions for display."""
    points = _validate_embedding(embedding)
    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3")
    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
        except ImportError as error:
            raise ImportError("t-SNE plotting requires 'FastMapy[plots]'.") from error
        kwargs.setdefault("perplexity", min(30, max(1, (len(points) - 1) / 3)))
        return TSNE(n_components=n_components, random_state=random_state, **kwargs).fit_transform(
            points
        )
    if method == "umap":
        try:
            import umap
        except ImportError as error:
            raise ImportError("UMAP plotting requires 'FastMapy[plots]'.") from error
        return umap.UMAP(
            n_components=n_components, random_state=random_state, **kwargs
        ).fit_transform(points)
    raise ValueError("method must be 'tsne' or 'umap'")


def plot_embedding(embedding, labels=None, color=None, dimensions=2, ax=None, show=True, **kwargs):
    """Plot a two- or three-dimensional embedding and return ``(figure, axes)``."""
    points = _validate_embedding(embedding)
    if dimensions not in (2, 3):
        raise ValueError("dimensions must be 2 or 3")
    if points.shape[1] != dimensions:
        raise ValueError("embedding dimensions must match the requested plot dimensions")
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("Plotting requires 'FastMapy[plots]'.") from error
    if ax is None:
        figure = plt.figure()
        ax = figure.add_subplot(projection="3d") if dimensions == 3 else figure.add_subplot()
    else:
        figure = ax.figure
    if dimensions == 2:
        ax.scatter(points[:, 0], points[:, 1], c=color, **kwargs)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, **kwargs)
    if labels is not None:
        if len(labels) != len(points):
            raise ValueError("labels must have one entry per embedded object")
        for point, label in zip(points, labels):
            ax.text(*point, str(label))
    if show:
        plt.show()
    return figure, ax


def _validate_embedding(embedding):
    points = np.asarray(embedding, dtype=float)
    if points.ndim != 2 or not len(points):
        raise ValueError("embedding must be a non-empty two-dimensional array")
    return points
