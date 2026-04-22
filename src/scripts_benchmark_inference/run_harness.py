# from src.scripts.build_instructions_db import load_all_instructions  # Import the instructions_db directly
from src.utils.args import parse_args
from src.utils.process_config import load_config
# from src.utils.io import get_brep_path_from_folder
import os
import os.path as osp

# import src.utils.rendering_utils as rendering_utils
import importlib
import json
# from src.utils.db import DatabaseManager

import pandas as pd
import argparse
import time
import shutil


task_instructions = {
    "edit": "Edit the existing model according to the provided instructions.",
}

def add_root_dir_to_files(task_dict, root_dir):
    import numpy as np
    
    extensions = ("png", "jpg", "jpeg", "f3d", "step", "smt", "stp", "stl", "mp4")
    
    def process_item(item):
        if isinstance(item, str) and item.endswith(extensions):
            return osp.join(root_dir, item)
        elif isinstance(item, dict):
            add_root_dir_to_files(item, root_dir)
        elif isinstance(item, (list, np.ndarray)):
            return [process_item(x) for x in item]
        return item
    
    for key, value in task_dict.items():
        task_dict[key] = process_item(value)

def format_task_dict(task_dict):
    # flatten the views list.
    # e.g. views: ['image1_sketch.png', 'image2_bottom.png'] -> view_sketch: 'image1_sketch.png, view_bottom: image2_bottom.png'
    orig_views = task_dict.pop("views", [])
    if orig_views:
        for view in orig_views:
            view_name = osp.splitext(osp.basename(view))[0].split("_")[-1]
            task_dict[f"view_{view_name}"] = view

    # process brep_start_path to create separate keys for each extension - usually f3d and step
    orig_brep_start = task_dict.pop("brep_start_path", [])
    if orig_brep_start:
        # Flatten nested lists (from converted numpy arrays)
        flat_paths = []
        for item in orig_brep_start:
            if isinstance(item, list):
                flat_paths.extend(item)
            else:
                flat_paths.append(item)
        
        for filename in flat_paths:
            extension = osp.splitext(osp.basename(filename))[1][1:]  # Get actual file extension
            task_dict[f"brep_start_path_{extension}"] = filename

    empty_strings = ["", "[]", "null"]
    # remove any keys with null values
    keys_to_remove = [key for key, value in task_dict.items() if not value or value in empty_strings]
    for key in keys_to_remove:
        task_dict.pop(key)

    return task_dict


# @ray.remote
def run_single_task(vlm, row, args):

    if isinstance(vlm, dict):
        vlm = load_model(vlm)
    
    row = row.to_dict()
    add_root_dir_to_files(row, args.db_base_path)
    row = format_task_dict(row)

    instructions = task_instructions.get(row["request_type"], "No instructions available.")

    start = time.time()
    edit_id = f"{args.userId}_{start}"

    sample_output_folder = os.path.join(args.output_folder, edit_id, "brep_end", str(start))
    os.makedirs(sample_output_folder, exist_ok=True)
    settings_output_fn = os.path.join(sample_output_folder, "settings.json")
    
    vlm_call_func = getattr(vlm, vlm.config["function"])
    vlm_call_func_kwargs = vlm.config.get("function_kwargs", {})

    success = False
    try:
    # if True:
        vlm_return_dict = vlm_call_func(
            instruction_text=instructions,
            task_info_dict=row,
            harness_script_file=args.harness,
            output_dir=sample_output_folder,
            **vlm_call_func_kwargs
        )
        success = True
    except Exception as e:
        print(f"Error processing request {row['request']}: {e}")
        vlm_return_dict = {}

    end = time.time()

    settings_dict = {
        "edit_request_id": row["request"],
        "edit_id": edit_id,
        "start_time": start,
        "end_time": end,
        "isHuman": False,
        "userId": args.userId,
        **vlm_return_dict,
    }

    # if the required extensions are not in the output folder, add a flag to the settings dict
    output_files = os.listdir(sample_output_folder)

    file_string = "tmp.step"
    filename = os.path.join(sample_output_folder, file_string) if file_string in output_files else None
    settings_dict["filename"] = filename

    if not success:
        settings_dict["failed_run"] = True

    if len(args.required_extensions) != 0:
        for ext in args.required_extensions:
            if not any(f.lower().endswith(f".{ext}") for f in output_files) :
                settings_dict["failed_run"] = True

    with open(settings_output_fn, "w") as f:
        json.dump(settings_dict, f, indent=4)

    if not success and args.remove_failed:
        shutil.rmtree(os.path.join(args.output_folder, edit_id), ignore_errors=True)

def process_parquet(args, config, model, required_extensions=[]):
    db_base_path = config["storage_dir"]["path"]
    # set args.db_base_path
    args.db_base_path = db_base_path
    output_folder = osp.join(args.output_dir, osp.splitext(osp.basename(args.harness))[0], args.userId, osp.splitext(osp.basename(args.input))[0])
    args.output_folder = output_folder

    parquet = pd.read_parquet(args.input)
    # rows = [x[1] for x in parquet.iterrows()]

    existing_request_ids = set()

    # crawl output folder, and check for any settings.json
    for root, dirs, files in os.walk(args.output_folder):
        for filename in files:
            if filename == "settings.json":
                with open(os.path.join(root, filename), "r") as f:
                    settings = json.load(f)
                    existing_request_ids.add(settings["edit_request_id"])

    print(existing_request_ids)

    # ray.init(num_cpus=4)
    # refs = [run_single_task.remote(model.config, row, args) for index, row in parquet.iterrows() if row["request"] not in existing_request_ids]
    # results = ray.get(refs)
    # ray.shutdown()
    done_count = 0
    for index, row in parquet.iterrows():
        if row["request"] in existing_request_ids:
            print(f"Skipping existing request {row['request']}. Already exists.")
            continue
        run_single_task(model, row, args)
        done_count += 1
        if done_count >= args.n_rows:
            break

    if args.remove_failed:
        cleanup_output_folder(output_folder, required_extensions=required_extensions)


def cleanup_output_folder(output_folder, required_extensions=[]):
    """
    Remove folders that don't have both a .step file and settings.json,
    but preserve any folder that is a subdirectory of a folder with valid files.
    Uses depth-first traversal.
    """
    if not os.path.exists(output_folder):
        return
    
    # Normalize the path
    output_folder = os.path.abspath(output_folder)
    
    def _has_valid_files(folder_path, required_extensions=required_extensions):
        """Check if folder has settings.json and files matching all required extensions."""
        try:
            files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

            required_exts = [ext.strip().lower() for ext in required_extensions]
            has_extensions = all(any(f.lower().endswith(f'.{ext}') for f in files) for ext in required_exts) if len(required_exts) > 0 else True
            has_settings = 'settings.json' in files
            return has_extensions and has_settings

        except (OSError, PermissionError):
            return False
    
    def _has_valid_ancestor(folder_path, root_path, required_extensions=required_extensions):
        """Check if any ancestor folder (up to root_path) has valid files."""
        current = folder_path
        while current != root_path and current != os.path.dirname(current):
            current = os.path.dirname(current)
            if _has_valid_files(current, required_extensions=required_extensions):
                return True
        return False
    
    def _check_folder(folder_path, required_extensions=required_extensions):
        """
        Recursively check folders depth-first.
        Returns True if this folder should be preserved, False otherwise.
        """
        if not os.path.isdir(folder_path):
            return False
        
        # Check if this folder has valid files
        has_valid_here = _has_valid_files(folder_path, required_extensions=required_extensions)
        
        # If this folder has valid files, preserve it and all its subdirectories
        if has_valid_here:
            return True
        
        # Check if any ancestor has valid files (if so, preserve this folder)
        if _has_valid_ancestor(folder_path, output_folder, required_extensions=required_extensions):
            return True
        
        # Recursively check all subdirectories (depth-first)
        has_valid_subdir = False
        try:
            entries = os.listdir(folder_path)
            subdirs = [e for e in entries if os.path.isdir(os.path.join(folder_path, e))]
            
            for entry in subdirs:
                entry_path = os.path.join(folder_path, entry)
                if _check_folder(entry_path, required_extensions=required_extensions):
                    has_valid_subdir = True
        except (OSError, PermissionError):
            return False
        
        # If this folder doesn't have valid files, no valid ancestors, and no valid subdirs, remove it
        if not has_valid_subdir:
            try:
                shutil.rmtree(folder_path)
                print(f"Removed folder: {folder_path}")
                return False  # This folder was removed
            except Exception as e:
                print(f"Error removing folder {folder_path}: {e}")
                return False
        
        # Return True if any subfolder should be preserved
        return has_valid_subdir
    
    # Start the recursive check, but don't remove the root folder itself
    # Only check and potentially remove subdirectories
    try:
        entries = os.listdir(output_folder)
        for entry in entries:
            entry_path = os.path.join(output_folder, entry)
            if os.path.isdir(entry_path):
                _check_folder(entry_path, required_extensions=required_extensions)
    except (OSError, PermissionError) as e:
        print(f"Error accessing folder {output_folder}: {e}")

def load_model(model_config,):
    # model_config = config["benchmark_models"][model_key]

    # use getattr to dynamically load the model class
    model_name = model_config["family"]
    module_path = f"src.vlms.{model_name}"
    module = importlib.import_module(module_path)
    model = module.VLM(model_config, cache=False)
    return model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the main db config file")
    parser.add_argument("--input", type=str, required=True, help="Input parquet file, or directory containing parquets")
    parser.add_argument("--userId", type=str, required=True, default="gemini-3-pro-thinking-high_cadquery-script")
    parser.add_argument("--harness", type=str, required=True, default="src/harnesses/cadquery_script.py", help="Path to the harness script")
    parser.add_argument("--output_dir", type=str, required=True, help="ouptut directory for model outputs")
    parser.add_argument("--required-extensions", type=str, nargs="*", default=["step"], help="List of required file extensions to consider a folder valid")
    parser.add_argument("--remove_failed", action="store_true", help="Whether to remove folders that don't have the required extensions")
    parser.add_argument("--n-rows", type=int, default=999999, help="Number of rows to process from the input parquet file")


    args = parser.parse_args()
    return args

def main():
    # Parse command-line arguments
    args = parse_args()
    # Load main configuration - required for VLM config
    config = load_config(args.config)
    model_config = config["benchmark_models"][args.userId]

    model = load_model(model_config)

    input_parquets = []

    if os.path.isdir(args.input):
        # process all parquet files in the directory
        for filename in os.listdir(args.input):
            if filename.endswith(".parquet") and "val" in filename:
                    input_parquets.append(os.path.join(args.input, filename))
    else:
        input_parquets.append(args.input)
    input_parquets.sort()

    for input_parquet in input_parquets:
        # check that model config "request_types" is empty, or contains the request type in the parquet filename
        request_type_in_filename = None
        for request_type in model.config.get("request_types", []):
            if request_type in osp.basename(input_parquet):
                request_type_in_filename = request_type
                break

        if model.config.get("request_types") and request_type_in_filename is None:
            print(f"Skipping {input_parquet} as it does not match request types for model {args.userId}")
            continue

        args.input = input_parquet
        process_parquet(args, config, model, required_extensions=args.required_extensions)


    



if __name__ == "__main__":
    main()