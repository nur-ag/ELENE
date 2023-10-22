import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("tables/results_scalability.csv")

# Introduce the necessary columns for analysis
df["num_nodes"] = (10 ** (df["T"] // 100 / 100)).round().astype(int)
df["max_degree"] = df["T"] % 100
df["depth"] = df.ELENE.str.split("-").apply(lambda x: int(x[1]))
df["model"] = df.ELENE.str.split("-").apply(lambda x: "ELENE ($\\bf{ND}$)" if x[2] == "YJN" else "ELENE ($\\bf{ED}$)" if x[0] != '0' else "GIN (Baseline)")
df["memory"] = (df.mem_mean).round().astype(int)

# Select the columns and prepare the plots
COLUMNS = ["num_nodes", "max_degree", "depth", "model", "memory"]
plot_df = df[COLUMNS]

# Tesla T4 - 15109MiB - 15.10GB limit
fig = plt.figure()
gs = fig.add_gridspec(3, hspace=0)
axes = gs.subplots(sharex=True)
fig.set_size_inches(8, 8)
powers_of_two = [0] + [2**i for i in range(4, 14)]
powers_of_ten = [10**i for i in np.arange(2, 4.51, 0.5)]
line_styles = {"GIN (Baseline)": ":", "ELENE ($\\bf{ND}$)": "-", "ELENE ($\\bf{ED}$)": "--"}
colors = ["k", "g", "b", "r"]
for index, grouping in enumerate(plot_df.groupby(["max_degree"])):
    ax = axes[index]
    max_deg, deg_df = grouping
    max_degree = max_deg[0]
    ax.set_xscale("log")
    ax.set_yscale("log", base=2, subs=[])
    if index == 1:
        ax.set_ylabel("Max. Memory Consumption (MB)")
    if index == 2:
        ax.set_xticks(ticks=powers_of_ten)
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
            linestyle=line_style
        )
        line_handles.append(handle)
    ax.legend(title=f"$d_{{max}}$ = {max_degree}", handles=line_handles, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_xlim([100, 10**4.5])
    ax.set_ylim([0, 16000])


fig.tight_layout()
fig.savefig(f"ScalabilityMaxDeg.pdf", dpi=120)


