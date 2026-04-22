import json
import os
import os.path as osp
from src.utils.args import parse_args
from src.utils.process_config import load_config
from src.utils.db import DatabaseManager
from importlib import import_module
from src.utils.visualise_results import all_tasks_radar_plot

def main():
    # Parse command-line arguments
    args = parse_args()
    # Load configuration
    config = load_config(args.config)

    dbm = DatabaseManager(config)

    dbm.clean_db_single_edit_per_user_per_request()

    all_results = {}

    # for benchmark in ["img2brep"]:
    for benchmark in config["benchmark_eval_users"]:

        print(f"Running benchmark: {benchmark}")

        file_str = f"src.scripts.benchmark_evals.{benchmark}"
        module = import_module(file_str)
        result_dict = module.run_benchmark_evals(dbm, config)
        for k, v in result_dict.items():
            all_results[k] = v

    # average together all results from human raters, where the key starts with "private." Then delete those individual results.
    # if there are any nans, nulls, Nones, ignore those entries in the average. But if all entries are nan, null, None, then return None and print a warning.
    # needs to handle the case where there are multiple human metrics, which are split by the last underscore, e.g. "private.12345_instruction" and "private.12345_quality".
    for task_key, task_results in all_results.items():
        for model_key, model_results in task_results.items():
            # Find all private.* keys and group by suffix (_instruction, _quality, etc.)
            private_keys_by_suffix = {}
            keys_to_delete = []
            
            for metric_key in list(model_results.keys()):
                if metric_key.startswith("private."):
                    keys_to_delete.append(metric_key)
                    # Extract suffix after last underscore (e.g., "_instruction", "_quality")
                    if "_" in metric_key:
                        suffix = metric_key.rsplit("_", 1)[-1]
                    else:
                        suffix = "rating"
                    
                    if suffix not in private_keys_by_suffix:
                        private_keys_by_suffix[suffix] = []
                    private_keys_by_suffix[suffix].append(metric_key)
            
            # Average each suffix group
            for suffix, metric_keys in private_keys_by_suffix.items():
                # Collect all edit_id -> values across all private raters
                edit_values = {}
                for metric_key in metric_keys:
                    for edit_id, value in model_results[metric_key].items():
                        if edit_id not in edit_values:
                            edit_values[edit_id] = []
                        # Only include valid values
                        if value is not None and value == value:  # value == value checks for NaN
                            edit_values[edit_id].append(value)
                
                # Compute average for each edit_id
                averaged_results = {}
                for edit_id, values in edit_values.items():
                    if values:
                        averaged_results[edit_id] = sum(values) / len(values)
                    else:
                        print(f"Warning: All values are None/NaN for {task_key}/{model_key}/human_{suffix}/{edit_id}")
                        averaged_results[edit_id] = None
                
                if averaged_results:
                    model_results[f"human_{suffix}"] = averaged_results
            
            # Delete original private.* keys
            for key in keys_to_delete:
                del model_results[key]




    # save results dict
    out_dir = osp.join(config["storage_dir"]["path"], "results")
    os.makedirs(out_dir, exist_ok=True)

    with open(osp.join(out_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=4)

    # Visualize results
    all_tasks_radar_plot(config=config, dbm=dbm, results=all_results)

if __name__ == "__main__":
    main()