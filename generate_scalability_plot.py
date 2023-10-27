import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("tables/results_scalability.csv")

# Introduce the necessary columns for analysis
df["num_nodes"] = (10 ** (df["T"] // 100 / 100)).round().astype(int)
df["max_degree"] = df["T"] % 100
df["depth"] = df.ELENE.str.split("-").apply(lambda x: int(x[1]))
df["model"] = df.ELENE.str.split("-").apply(lambda x: "ELENE ($\\bf{ND}$)" if x[2] == "YJN" else "ELENE ($\\bf{ED}$)" if x[0] != '0' else "GIN (Lower Bound)")
df["memory"] = (df.mem_mean).round().astype(int)

# Select the columns and prepare the plots
COLUMNS = ["num_nodes", "max_degree", "depth", "model", "memory"]
plot_df = df[COLUMNS]

# Tesla T4 - 15109MiB - 15.10GB limit
powers_of_two = [0] + [2**i for i in range(4, 14)]
powers_of_ten = [10**i for i in np.arange(2, 4.51, 0.25)]
power_of_ten_labels = [f"$10^{int(i)}$" if i == int(i) else f"$10^{{{i}}}$" for i in np.arange(2, 4.51, 0.25)]
line_styles = {"GIN (Lower Bound)": ":", "ELENE ($\\bf{ND}$)": "-", "ELENE ($\\bf{ED}$)": "--"}
colors = ["k", "g", "b", "r"]
markers = [".", ".", ".", "."]
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
        key=lambda x: (x[0][0], "ND" not in x[0][1])
    )
    for group, group_df in ordered_grouping: 
        depth, model = group
        line_style = line_styles[model]
        model_str = model if not depth else f"k={depth}, {model}"
        handle, = ax.plot(
            group_df.num_nodes, 
            group_df.memory, 
            c=colors[depth], 
            label=model_str,
            linestyle=line_style,
            marker=markers[depth],
        )
        line_handles.append(handle)
    ax.legend(title=f"$d_{{max}}$ = {max_degree}", handles=line_handles, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_xlim([10**1.95, 10**4.55])
    ax.set_ylim([0, 16000])
    fig.tight_layout()
    fig.savefig(f"ScalabilityMaxDeg{max_degree}.pdf", dpi=120)
