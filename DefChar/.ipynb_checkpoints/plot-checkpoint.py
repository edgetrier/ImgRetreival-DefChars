from matplotlib import pyplot as plt
import matplotlib as mpl
import random

def hue_bar_avg_md(ax, rang, range_stable, stable_warn=0.05, dc_name=None, overall=None, count=None, score=None):
    
    color_bar = [mpl.cm.Purples, mpl.cm.Blues, mpl.cm.Greens, mpl.cm.Oranges, mpl.cm.Reds, mpl.cm.YlOrRd, mpl.cm.OrRd]
colour_choice = color_bar[random.randint(0, len(color_bar)-1)]
    x_labels = ['', 'red', 'orange', 'yellow', 'lime', 'green', "light-green", 'cyan', 'light-blue', 'blue', 'purple', 'violet', 'pink', "red"]

    # Calculate Bound
    left = rang[0] * 360 - 1
    right = rang[1] * 360 - 1
    left_s = range_stable[0] * 360 -1
    right_s = range_stable[1] * 360 -1
    left_style = "-"
    right_style = "-"
    if abs(range_stable[0]) >= stable_warn:
        left_style = ":"
    if abs(range_stable[1]) >= stable_warn:
        right_style = ":"

    # Draw Colour Bar
    c = [[i for i in range(360)] for _ in range(20)]
    ax.imshow(c, cmap=mpl.cm.hsv)

    # Left Bound
    ax.plot([left,left], [-1,20], c="k", linewidth=15, linestyle=left_style)
    ax.plot([left,left+5], [-1,-1], c="k", linewidth=15, linestyle=left_style)
    ax.plot([left,left+5], [20,20], c="k", linewidth=15, linestyle=left_style)

    # Right Bound
    ax.plot([right,right], [-1,20], c="k", linewidth=15, linestyle=right_style)
    ax.plot([right,right-5], [-1,-1], c="k", linewidth=15, linestyle=right_style)
    ax.plot([right,right-5], [20,20], c="k", linewidth=15, linestyle=right_style)

    # Two Area
    if left > right:
        ax.plot([359,359], [-1,20], c="k", linewidth=15, linestyle="-")
        ax.plot([359,354], [-1,-1], c="k", linewidth=15, linestyle="-")
        ax.plot([359,354], [20,20], c="k", linewidth=15, linestyle="-")

        ax.plot([0,0], [-1,20], c="k", linewidth=15, linestyle="-")
        ax.plot([0,5], [-1,-1], c="k", linewidth=15, linestyle="-")
        ax.plot([0,5], [20,20], c="k", linewidth=15, linestyle="-")

    # Left Arrow
    ax.arrow(left, 5, left_s, 0, width=1, color="k", length_includes_head=True)

    # Right Arrow
    ax.arrow(right, 15, right_s, 0, width=1, color="k", length_includes_head=True)

    #Title
    Title = ""
    if dc_name is not None:
        Title += dc_name + "\n"
    if overall is not None:
        Title += "Overall Score: " + str(overall)
    if score is not None:
        Title += "Importance Score: " + str(score)
    if count is not None:
        Title += "Counts per Tree: " + str(count)

    ax.set_title(Title, fontsize=100, fontweight="heavy", y=1.2)

    # Setting
    ax.plot([-1,360], [-3,-3], visible=False)
    ax.plot([-1,360], [22,22], visible=False)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(14))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticklabels(x_labels, fontsize=80)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    ax.tick_params(axis='x', which='both', width=15, size=30, colors="dimgrey", pad=30)
    
    return ax


def hue_bar_range(ax, rang, range_stable, stable_warn=0.05, overall=None, count=None, score=None):
    pass