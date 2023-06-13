import glob
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Datasets in our complete benchmark
TU_DATASETS = ["MUTAG", "PTC", "PROTEINS", "NCI1"]
OGBG_DATASETS = ["ogbg-molhiv", "ogbg-molpcba"]
BENCH = ["ZINC", "PATTERN", "CIFAR10"]
EXPR = ["exp-classify", "sr25-classify", "count_substructures", "graph_property"]
PROX = ["Proximity"]
QM9 = ["qm9"]

# Dataset result perf. metric formats
USES_PERCENT = ["Proximity", "exp-clasify", "sr25-classify", "CIFAR10", "ogbg-molhiv"] + TU_DATASETS
BENCHMARKS_BY_PRECISION = {
    "expressivity": 3,
    "benchmark": 3,
    "proximity": 1,
    "tu": 1,
    "qm9": 3,
}

# Datasets that we use for the tables in the paper
FINAL_DATASETS = OGBG_DATASETS + BENCH + EXPR + PROX + QM9
DATASETS_BY_NAME = {
    "expressivity": EXPR,
    "proximity": PROX,
    "benchmark": BENCH + OGBG_DATASETS,
    "tu": TU_DATASETS,
    "qm9": QM9,
}

# Analysis constants
METRIC_STATS = ["mean", "std", "count"]
EXPERIMENT_KEY = ["dataset", "T", "GNN", "Hops", "Emb", "Mini", "IgA", "IgR", "IgE", "EIGEL"]
ORDERING_KEY = ["dataset", "T", "GNN", "Hops", "Emb", "IsGNNAK", "IsIGEL", "IsEIGEL", "IsEIGELL", "EIGELLearnType", "FullModel", "Model"]
REPORTING_KEY = ["dataset", "T"]

# Parse models in simple / full formats
def parse_model(row, is_full=False):
    model = "GIN-AK" if row.IsGNNAK else "GIN"
    if row.IsEIGEL:
        format = f" (k = {row.IgA})" if is_full else ""
        return f"{model}+EIGEL{format}"
    elif row.IsIGEL:
        format = f" (k = {row.IgA})" if is_full else ""
        return f"{model}+IGEL{format}"
    elif row.IsEIGELL:
        format = f" ({row.EIGELLearnType})" if is_full else ""
        return f"{model}+EIGEL-L{format}"
    return model

DATASET_FORMATS = {
    "expressivity": "FullModel",
    "proximity": "Model",
    "benchmark": "Model",
    "tu": "Model",
    "qm9": "Model",
}

# Load all the Tensorboard logs directly into Pandas
combined_df_data = []
all_experiment_log_dirs = glob.glob("results/final/*/*/")
for log_dir in all_experiment_log_dirs:
    event_accumulator = EventAccumulator(log_dir)
    event_accumulator.Reload()
    dataset_exp = log_dir.split("/")[2:]
    dataset = dataset_exp[0]
    experiment_params = dataset_exp[1][:-1].split("] ")
    experiment_params = [exp.strip().split("[") for exp in experiment_params]
    experiment_params = {param: value for param, value in experiment_params}
    for scalar in event_accumulator.scalars.Keys():
        scalar_tokens = scalar.split("/")
        run = int(scalar_tokens[0][-1])
        metric = scalar_tokens[-1]
        events = event_accumulator.Scalars(scalar)
        num_steps = max([x.step for x in events])
        datapoints = [x.value for x in events]
        last_datapoint = datapoints[-1]
        data_dict = {
            "dataset": dataset,
            "run": run,
            "metric": metric,
            "num_steps": num_steps,
            "datapoints": datapoints,
            "last_datapoint": last_datapoint,
            **experiment_params
        }
        combined_df_data.append(data_dict)


# Pull the best result per model type, dataset, task on both test set performance & memory
df = pd.DataFrame.from_dict(combined_df_data)
test_df = df[df.metric == "test-best-perf"]

# Scale QM9 results and rename tasks for display
from train.qm9 import TASKS as QM9_TASKS, CHEMICAL_ACC_NORMALISING_FACTORS
test_df["last_datapoint"] = [
    row["last_datapoint"] / (1 if row.dataset != "qm9" else CHEMICAL_ACC_NORMALISING_FACTORS[int(row["T"])])
    for i, row in test_df.iterrows()
]

mem_df = df[df.metric == "memory"]
def aggregate_data(metric_df, experiment_key=EXPERIMENT_KEY, metric_stats=METRIC_STATS):
    results_df = metric_df.groupby(
        experiment_key
    ).agg({"last_datapoint": metric_stats}).reset_index()
    results_df.columns = [top if not bottom else bottom for (top, bottom) in results_df.columns]
    return results_df

def best_results_df(metric_df, ordering_key=ORDERING_KEY):
    test_results_df = metric_df.copy()
    test_results_df["IsGNNAK"] = test_results_df.Mini.astype(int) > 0
    test_results_df["IsIGEL"] = test_results_df.IgA.astype(int) > 0
    test_results_df["IsEIGEL"] = test_results_df.IsIGEL & (test_results_df.IgR == "True")
    test_results_df["EIGELLearnType"] = ["" if eigel.startswith("0-0-") else eigel.split("-")[2] for eigel in test_results_df.EIGEL]
    test_results_df["IsEIGELL"] = test_results_df.EIGELLearnType != ""
    test_results_df["FullModel"] = [parse_model(row, True) for i, row in test_results_df.iterrows()]
    test_results_df["Model"] = [parse_model(row, False) for i, row in test_results_df.iterrows()]
    best_mean_df = test_results_df.groupby(ordering_key).agg(mean=("mean", "max")).reset_index()
    return pd.merge(test_results_df, best_mean_df, on=ordering_key + ["mean"])


# Rename the columns and merge both tables
best_test_df = best_results_df(aggregate_data(test_df))
best_test_df.columns = [col if col not in METRIC_STATS else f"btest_{col}" for col in best_test_df.columns]
match_mem_df = aggregate_data(mem_df)
match_mem_df.columns = [col if col not in METRIC_STATS else f"mem_{col}" for col in match_mem_df.columns]
match_mem_df = match_mem_df[EXPERIMENT_KEY + [col for col in match_mem_df.columns if col.startswith("mem_")]]

# Join the best results with their memory budget and organize the results for visualization
best_df = pd.merge(best_test_df, match_mem_df, on=EXPERIMENT_KEY)
best_col_order = [col for i, col in sorted(enumerate(best_df.columns), key=lambda x: (x[1].split("_")[-1] in METRIC_STATS, x[0]))]
full_best_df = best_df[best_col_order].copy()
full_best_df.to_csv(f"tables/results_all.csv")

# Get the row with the best memory usage and best performance
for benchmark, datasets in DATASETS_BY_NAME.items():
    reporting_key = [*REPORTING_KEY] + [DATASET_FORMATS[benchmark]]
    best_df = full_best_df.copy()
    best_df = best_df[best_df.dataset.isin(datasets)]
    best_perf_df = best_df.groupby(reporting_key).agg(btest_mean=("btest_mean", "max")).reset_index()
    best_df = pd.merge(best_df, best_perf_df, on=reporting_key + ["btest_mean"])
    best_mem_df = best_df.groupby(reporting_key).agg(mem_mean=("mem_mean", "min")).reset_index()
    best_df = pd.merge(best_df, best_mem_df, on=reporting_key + ["mem_mean"])

    # Adjust reporting and round up
    dataset_reporting_factor = [100.0 if dataset in USES_PERCENT else 1.0 for dataset in best_df.dataset]
    best_df.btest_mean *= dataset_reporting_factor
    best_df.btest_std *= dataset_reporting_factor

    rounding_precision = BENCHMARKS_BY_PRECISION[benchmark]
    best_df.btest_mean = best_df.btest_mean.round(rounding_precision)
    best_df.btest_std = best_df.btest_std.round(rounding_precision)

    # Report the task as string
    if benchmark == "qm9":
        best_df["T"] = [QM9_TASKS[int(task)] for task in best_df["T"]]

    # Dump the results to separate files
    best_df.to_csv(f"tables/results_{benchmark}.csv")

