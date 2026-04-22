from src.utils.args import parse_args
from src.utils.process_config import load_config
import os
import os.path as osp
import json
from src.utils.db import DatabaseManager
import pandas as pd

def create_train_parquet(config: dict, dbm: DatabaseManager):
    # get all edits from the database
    all_edits_iterator = dbm.edits.find()

    rows = []

    # loop over all edits and create a parquet file with all the edits
    for edit in all_edits_iterator:
        request = dbm.requests.find_one({"_id": edit["request"], "user": edit["user"]})

        if not request:
            print(f"Skipping edit {edit['_id']} because request not found")
            continue

        brep_start_id = request["brep_start"]
        brep_end_id = edit["brep_end"]

        brep_start = dbm.breps.find_one({"_id": brep_start_id})
        brep_end = dbm.breps.find_one({"_id": brep_end_id})

        if brep_start is None or brep_end is None:
            print(f"Skipping edit {edit['_id']} because brep_start or brep_end is None")
            continue

        brep_start_path = brep_start["f3d"][0]
        brep_end_path = brep_end["f3d"][0]

        # add to parquet file
        row = {
            "edit": edit["_id"],
            "request": edit["request"],
            "file_name": request.get("filename", None),
            "brep_start_path": brep_start_path,
            "request_video_30fps_720p": request["30_720"],
            "request_transcript": json.dumps(request.get("transcript", None)),
            "request_events": json.dumps(request["events"]),
            "request_text": request.get("text", None),
            "edit_events": json.dumps(edit["events"]),
            "edit_frames_dir": osp.join(brep_end_path, "frames"),
            "brep_end_path": brep_end_path,
        }

        rows.append(row)

    out_dir = osp.join(config["storage_dir"]["path"], "parquets")
    os.makedirs(out_dir, exist_ok=True)
    out_fn = osp.join(out_dir, "train.parquet")
    df = pd.DataFrame(rows)
    df.to_parquet(out_fn, index=False)

def create_val_tasks_parquet(config: dict, dbm: DatabaseManager, request_type=None, benchmark_type=None, views=[], copy_files=False, request_ids=None, n_to_save=None):
    # get all edits from the database


    find_dict = {}
    if request_type:
        find_dict["request_type"] = request_type
    if benchmark_type:
        find_dict[benchmark_type] = True

    all_requests_iterator = dbm.requests.find(find_dict)

    rows = []

    # loop over all edits and create a parquet file with all the edits
    requests_saved = 0
    for request in all_requests_iterator:

        if request_ids and request["_id"] not in request_ids:
            continue

        if n_to_save and requests_saved >= n_to_save:
            break

        brep_start_id = request.get("brep_start", None)

        brep_start_path = []
        if brep_start_id:
            brep_start = dbm.breps.find_one({"_id": brep_start_id})
            for ext_str in ["f3d", "step"]:
                if ext_str in brep_start:
                    brep_start_path.append(brep_start[ext_str])

        request_text = request.get("text", None)
        if not request_text:
            request_text = request.get("prompt", None)

        row = {
            "request": request["_id"],
            "request_type": request["request_type"],
            "file_name": request.get("filename", None),
            "brep_start_path": brep_start_path if brep_start_path else None,
            "request_video_30fps_720p_audio": request.get("30_720_audio", None),
            "request_transcript": json.dumps(request.get("transcript", None)),
            "request_events": json.dumps(request["events"]),
            "request_text": request_text,
            "views": dbm.get_brep_images(brep_start_id, format=["png", "jpg"], views=views) if brep_start_id else None,
        }

        # uncomment to also include some model outputs in the parquet for comparison. Only use for testing.
        # gt brep
        gen_model_users = ["gemini-3-pro-thinking-high_freecad-script", "gemini-3-flash_freecad-script", "gpt-5.2_freecad_script", "claude-sonnet-4.5_freecad_script"]
        gen_model_dict = {u: None for u in gen_model_users}
        for gen_model_user in gen_model_dict.keys():
            edit = dbm.edits.find_one({"request": request["_id"], "user": gen_model_user})
            if not edit:
                continue
            brep_end_id = edit.get("brep_end", None)
            if not brep_end_id:
                continue
            brep_end = dbm.breps.find_one({"_id": brep_end_id})
            if not brep_end:
                continue
            brep_end_step = brep_end.get("step", None)
            gen_model_dict[gen_model_user] = brep_end_step

        row.update(gen_model_dict)

        gt_brep = None
        # find gt brep. The edit and request have the same user
        edit = dbm.edits.find_one({"request": request["_id"], "user": request["user"]})
        if edit:
            brep_end_id = edit.get("brep_end", None)
            if brep_end_id:
                brep_end = dbm.breps.find_one({"_id": brep_end_id})
                if brep_end:
                    gt_brep = brep_end.get("step", None)
        row["gt_brep_path"] = gt_brep

        rows.append(row)
        requests_saved += 1

    out_dir = osp.join(config["storage_dir"]["path"], "parquets_victor_2")
    os.makedirs(out_dir, exist_ok=True)
    if benchmark_type:
        out_fn = osp.join(out_dir, f"val_{request_type}_{benchmark_type}.parquet")
    else:
        out_fn = osp.join(out_dir, f"val_{request_type}_all.parquet")
    df = pd.DataFrame(rows)
    df.to_parquet(out_fn, index=False)

    # uncomment to copy files to ouptut directory. Only use for testing.

    if not copy_files:
        return
    

    extensions = [".f3d", ".step", ".jpg", ".png", ".mp4"]
    for row in rows:
        for key, value in row.items():
            # if a value is a filepath, append the filepath to the out_dir, and copy the file to that location
            # it should handle nested filepaths, and lists of filepaths
            if value is None:
                continue
            if isinstance(value, str):
                for ext in extensions:
                    if value.endswith(ext):
                        src_path =  osp.join(config["storage_dir"]["path"], value)
                        if not osp.exists(src_path):
                            print(f"File {src_path} does not exist, skipping copy.")
                            continue
                        dst_path = osp.join(out_dir, value)
                        # make directories if they don't exist
                        os.makedirs(osp.dirname(dst_path), exist_ok=True)
                        if not osp.exists(dst_path):
                            os.system(f"cp '{src_path}' '{dst_path}'")
            elif isinstance(value, list):
                # Flatten nested lists
                flat_items = []
                def flatten(lst):
                    for item in lst:
                        if isinstance(item, list):
                            flatten(item)
                        else:
                            flat_items.append(item)
                flatten(value)
                for item in flat_items:
                    print(item)
                    if not isinstance(item, str):
                        continue
                    for ext in extensions:
                        if item.endswith(ext):
                            src_path =  osp.join(config["storage_dir"]["path"], item)
                            if not osp.exists(src_path):
                                print(f"File {src_path} does not exist, skipping copy.")
                                continue
                            dst_path = osp.join(out_dir, item)
                            # make directories if they don't exist
                            os.makedirs(osp.dirname(dst_path), exist_ok=True)
                            if not osp.exists(dst_path):
                                os.system(f"cp '{src_path}' '{dst_path}'")



def main():
    # Parse command-line arguments
    args = parse_args()
    # Load configuration
    config = load_config(args.config)

    dbm = DatabaseManager(config)
    create_val_tasks_parquet(config, dbm, request_type="edit")
        

if __name__ == "__main__":
    main()