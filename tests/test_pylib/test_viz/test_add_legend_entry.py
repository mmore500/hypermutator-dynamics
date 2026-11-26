import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

from pylib.viz._add_legend_entry import add_legend_entry

df = pd.DataFrame(
    {"x": [0, 1, 0, 1], "y": [0, 1, 1, 0], "what": ["a", "a", "b", "b"]}
)


def test_add_legend_entry_axes():
    __, ax = plt.subplots()
    sns.lineplot(df, x="x", y="y", hue="what", ax=ax)
    legend = add_legend_entry(ax)
    assert legend is ax.get_legend()
    assert [t.get_text() for t in legend.texts] == [
        "a",
        "b",
        "analytic approximation",
    ]
    assert legend.legend_handles[-1].get_linestyle() == "--"
    assert legend.get_title().get_text() == "what"
    plt.close("all")


def test_add_legend_entry_grid_keeps_position():
    grid = sns.relplot(df, x="x", y="y", hue="what", kind="line")
    sns.move_legend(
        grid, "lower center", bbox_to_anchor=(0.4, 0.9), ncol=2, title=None
    )
    grid.figure.canvas.draw()
    before = grid.legend.get_window_extent()
    legend = add_legend_entry(grid, "analytic\napprox")
    assert legend is grid.legend
    assert len(grid.figure.legends) == 1
    assert legend._ncols == 3
    assert [t.get_text() for t in legend.texts][-1] == "analytic\napprox"
    grid.figure.canvas.draw()
    after = legend.get_window_extent()
    assert abs(after.y0 - before.y0) < 1
    assert abs((after.x0 + after.x1) - (before.x0 + before.x1)) < 2
    plt.close("all")


def test_add_legend_entry_no_legend():
    __, ax = plt.subplots()
    assert add_legend_entry(ax) is None
    plt.close("all")


def test_add_legend_entry_before():
    __, ax = plt.subplots()
    sns.lineplot(df, x="x", y="y", hue="what", ax=ax)
    legend = add_legend_entry(ax, before="b")
    assert [t.get_text() for t in legend.texts] == [
        "a",
        "analytic approximation",
        "b",
    ]
    assert legend.legend_handles[1].get_linestyle() == "--"
    plt.close("all")
