import glob
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TU_DATASETS = ["MUTAG", "PTC", "PROTEINS", "NCI1"]
OGBG_DATASETS = ["ogbg-molhiv", "ogbg-molpcba"]
BENCH = ["ZINC", "PATTERN", "CIFAR10"]
EXPR = ["exp-classify", "sr25-classify", "count_substructures", "graph_property"]
PROX = ["Proximity"]

FINAL_DATASETS = OGBG_DATASETS + BENCH + EXPR + PROX


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




df = pd.DataFrame.from_dict(combined_df_data)
df = df[df.dataset.isin(FINAL_DATASETS)]
EXPERIMENT_KEY = ["dataset", "T", "GNN", "Emb", "Mini", "IgA", "IgR", "IgE", "EIGEL"]
test_results_df = df[df.metric == "test-best-perf"].groupby(
    EXPERIMENT_KEY
).agg({"last_datapoint": ["mean", "std", "count"]}).reset_index()
test_results_df.columns = [top if not bottom else bottom for (top, bottom) in test_results_df.columns]
test_results_df["IsGNNAK"] = test_results_df.Mini.astype(int) > 0
test_results_df["IsIGEL"] = test_results_df.IgA.astype(int) > 0
test_results_df["IsEIGEL"] = test_results_df.IsIGEL & (test_results_df.IgR == "True")
test_results_df["EIGELLearnType"] = ["" if eigel.startswith("0-0-") else eigel.split("-")[2] for eigel in test_results_df.EIGEL]
ORDERING_KEY = ["dataset", "T", "GNN", "Emb", "IsGNNAK", "IsIGEL", "IsEIGEL", "EIGELLearnType"]
best_mean_df = test_results_df.groupby(ORDERING_KEY).agg(mean=("mean", "max")).reset_index()
best_rest_results_df = pd.merge(test_results_df, best_mean_df, on=ORDERING_KEY + ["mean"])
