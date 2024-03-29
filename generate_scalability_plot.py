import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

DATASET = "kregular"
df = pd.read_csv("tables/results_scalability.csv")
df = df[df.dataset == DATASET]

# Introduce the necessary columns for analysis
df["num_nodes"] = (10 ** (df["T"] // 100 / 100)).round().astype(int)
df["max_degree"] = df["T"] % 100
df["depth"] = df.Hops.apply(int)
df["model"] = df.ELENE.str.split("-").apply(lambda x: "ELENE ($\\bf{ND}$)" if x[2] == "YJN" else "ELENE ($\\bf{ED}$)" if x[0] != '0' else "Baseline")
df["model"] = [row["model"] if row["model"] != "Baseline" else row["Model"] for i, row in df.iterrows()]
df["model"] = ["GIN (Baseline)" if row["model"] == "GIN" else row["model"] for i, row in df.iterrows()]
df["memory"] = (df.mem_mean).round().astype(int)

# Select the columns and prepare the plots
COLUMNS = ["num_nodes", "max_degree", "depth", "model", "memory"]
plot_df = df[COLUMNS]

# Tesla T4 - 15109MiB - 15.10GB limit
powers_of_two = [0] + [2**i for i in range(4, 14)]
powers_of_ten = [10**i for i in np.arange(2, 4.51, 0.25)]
power_of_ten_labels = [f"$10^{int(i)}$" if i == int(i) else f"$10^{{{i}}}$" for i in np.arange(2, 4.51, 0.25)]
line_styles = {"GIN (Baseline)": ":", "GIN-AK": ":", "GIN-AK+": ":", "ELENE ($\\bf{ND}$)": "-", "ELENE ($\\bf{ED}$)": "--"}
colors = ["k", "g", "b", "r"]
markers = {
    "GIN (Baseline)": (".", 4.0),
    "GIN-AK": ("x", 3.0),
    "GIN-AK+": ("+", 3.0),
    "ELENE ($\\bf{ND}$)": (".", 4.0),
    "ELENE ($\\bf{ED}$)": ("s", 3.0)
}
for max_deg, deg_df in plot_df.groupby(["max_degree"]):
    fig = plt.figure()
    fig.set_size_inches(8, 3)
    gs = fig.add_gridspec()
    ax = gs.subplots()
    ax.grid(True, alpha=0.4, linewidth=0.5)

    # Parse the max degree
    max_degree, = max_deg

    # Set ax information
    ax.set_xscale("log", subs=[])
    ax.set_yscale("log", base=2, subs=[])
    ax.set_ylabel("Max. Memory Consumption (MB)")
    ax.set_xticks(ticks=powers_of_ten, labels=power_of_ten_labels)
    ax.set_xlabel("Graph Size (Nr. of Nodes)")
    ax.set_yticks(ticks=powers_of_two, labels=[f"{int(p)}MB" for p in powers_of_two])

    line_handles = []
    ax_line = ax.axhline(y=15109, color="k", linestyle='-.', lw=1.0, label="Max. GPU Memory")
    line_handles.append(ax_line)
    ordered_grouping = sorted(
        deg_df.groupby(["depth", "model"]),
        key=lambda x: (x[0][0], "GIN" not in x[0][1], "ND" not in x[0][1])
    )
    for group, group_df in ordered_grouping:
        depth, model = group
        line_style = line_styles[model]
        model_str = model if not depth else (f"{model}" + " " * 10)
        handle, = ax.plot(
            group_df.num_nodes,
            group_df.memory,
            c=colors[depth],
            label=model_str,
            linestyle=line_style,
            alpha=None if "GIN" not in model_str else 0.6,
            marker=markers[model][0],
            markersize=markers[model][1],
            linewidth=None if "GIN-AK" not in model_str else 1.0,
        )
        line_handles.append(handle)
    # Ugly: draw 4 legends stacked on top of each other to show 'categories' at each value of k plus the header with baselines
    legend1 = ax.legend(title="$\\mathbf{k=1}$:" + " " * 21, handles=line_handles[2:6], loc="center left", bbox_to_anchor=(1, 0.675), frameon=False, prop={'size': 5}, title_fontsize=6)
    legend1.get_title().set_color(colors[1])
    legend2 = ax.legend(title="$\\mathbf{k=2}$:" + " " * 21, handles=line_handles[6:10], loc="center left", bbox_to_anchor=(1, 0.4), frameon=False, prop={'size': 5}, title_fontsize=6)
    legend2.get_title().set_color(colors[2])
    legend3 = ax.legend(title="$\\mathbf{k=3}$:" + " " * 21, handles=line_handles[10:14], loc="center left", bbox_to_anchor=(1, 0.125), frameon=False, prop={'size': 5}, title_fontsize=6)
    legend3.get_title().set_color(colors[3])
    legend4 = ax.legend(title=f"$d_{{max}}$ = {max_degree}", handles=line_handles[:2], loc="center left", bbox_to_anchor=(1, 0.91), frameon=False, prop={'size': 5})
    fig.add_artist(legend1)
    fig.add_artist(legend2)
    fig.add_artist(legend3)
    fig.add_artist(legend4)
    ax.set_xlim([10**1.95, 10**4.55])
    ax.set_ylim([0, 16000])
    fig.tight_layout()
    fig.savefig(f"ScalabilityMaxDeg{max_degree}.pdf", dpi=120)


NUM_NODES = int(10**3.0)
subset_df = plot_df[plot_df.num_nodes == NUM_NODES]
fig = plt.figure()
fig.set_size_inches(8, 3)
gs = fig.add_gridspec()
ax = gs.subplots()
ax.grid(True, alpha=0.4, linewidth=0.5)

# Set ax information
ax.set_yscale("log", base=2, subs=[])
ax.set_ylabel("Max. Memory Consumption (MB)")
ax.set_xlabel("$d_{{max}}$ (Max. Degree)")
ax.set_xticks(ticks=[6, 12, 18, 24, 30, 36])
ax.set_yticks(ticks=powers_of_two, labels=[f"{int(p)}MB" for p in powers_of_two])

line_handles = []
ax_line = ax.axhline(y=15109, color="k", linestyle='-.', lw=1.0, label="Max. GPU Memory")
line_handles.append(ax_line)
ordered_grouping = sorted(
    subset_df.groupby(["depth", "model"]),
    key=lambda x: (x[0][0], "GIN" not in x[0][1], "ND" not in x[0][1])
)
for group, group_df in ordered_grouping:
    depth, model = group
    line_style = line_styles[model]
    model_str = model if not depth else (f"{model}" + " " * 10)
    handle, = ax.plot(
        group_df.max_degree,
        group_df.memory,
        c=colors[depth],
        label=model_str,
        linestyle=line_style,
        alpha=None if "GIN" not in model_str else 0.6,
        marker=markers[model][0],
        markersize=markers[model][1],
        linewidth=None if "GIN-AK" not in model_str else 1.0,
    )
    line_handles.append(handle)
# Ugly: draw 4 legends stacked on top of each other to show 'categories' at each value of k plus the header with baselines
legend1 = ax.legend(title="$\\mathbf{k=1}$:" + " " * 21, handles=line_handles[2:6], loc="center left", bbox_to_anchor=(1, 0.675), frameon=False, prop={'size': 5}, title_fontsize=6)
legend1.get_title().set_color(colors[1])
legend2 = ax.legend(title="$\\mathbf{k=2}$:" + " " * 21, handles=line_handles[6:10], loc="center left", bbox_to_anchor=(1, 0.4), frameon=False, prop={'size': 5}, title_fontsize=6)
legend2.get_title().set_color(colors[2])
legend3 = ax.legend(title="$\\mathbf{k=3}$:" + " " * 21, handles=line_handles[10:14], loc="center left", bbox_to_anchor=(1, 0.125), frameon=False, prop={'size': 5}, title_fontsize=6)
legend3.get_title().set_color(colors[3])
legend4 = ax.legend(title=f"$N$ = {NUM_NODES}", handles=line_handles[:2], loc="center left", bbox_to_anchor=(1, 0.91), frameon=False, prop={'size': 5})
fig.add_artist(legend1)
fig.add_artist(legend2)
fig.add_artist(legend3)
fig.add_artist(legend4)
ax.set_xlim([5, 37])
ax.set_ylim([0, 16000])
fig.tight_layout()
fig.savefig(f"ScalabilityMaxDegMemory.pdf", dpi=120)
