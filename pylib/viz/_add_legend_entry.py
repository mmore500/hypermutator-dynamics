import typing

from matplotlib import axes as mpl_axes
from matplotlib import legend as mpl_legend
from matplotlib import lines as mpl_lines
import seaborn as sns


def add_legend_entry(
    obj: typing.Union[sns.FacetGrid, mpl_axes.Axes],
    label: str = "analytic approximation",
    **kwargs,
) -> typing.Optional[mpl_legend.Legend]:
    """Append a line entry to the existing legend of a seaborn grid or
    matplotlib axes, keeping the legend's position and styling.

    Keyword arguments set the entry's line properties; the default is a
    dashed black line, as used for math model predictions. Returns the new
    legend, or None if `obj` has no legend.
    """
    is_grid = not isinstance(obj, mpl_axes.Axes)
    old = obj.legend if is_grid else obj.get_legend()
    if old is None:
        return None

    handles = [*old.legend_handles]
    labels = [text.get_text() for text in old.texts]
    handles.append(
        mpl_lines.Line2D([], [], **{"color": "black", "ls": "--", **kwargs})
    )
    labels.append(label)

    owner = old.axes if old.isaxes else old.figure
    transform = old.axes.transAxes if old.isaxes else old.figure.transFigure
    legend_kws = dict(
        loc=old._loc,
        frameon=old.get_frame_on(),
        # keep single-row legends in a single row
        ncols=old._ncols + (1 < old._ncols >= len(old.texts)),
        columnspacing=old.columnspacing,
        title=old.get_title().get_text() or None,
    )
    if old.texts:
        legend_kws["fontsize"] = old.texts[0].get_fontsize()
    if old._bbox_to_anchor is not None:
        legend_kws["bbox_to_anchor"] = old.get_bbox_to_anchor().transformed(
            transform.inverted()
        )
        legend_kws["bbox_transform"] = transform

    old.remove()
    new = owner.legend(handles, labels, **legend_kws)
    new.get_title().set_fontsize(old.get_title().get_fontsize())
    if is_grid:
        obj._legend = new
    return new
