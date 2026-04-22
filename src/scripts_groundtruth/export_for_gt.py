from src.utils.args import parse_args
from src.utils.process_config import load_config
import os
import json
from src.utils.db import DatabaseManager
import shutil


def export_files(config: dict, dbm: DatabaseManager, output_folder_relative: str, s3_base_path: str, task_dict: dict={}, models: list=[], n_ratings_threshold: int=999, ignore_failed_runs: bool=True) -> None:
    
    output_folder = os.path.join(config["storage_dir"]["path"], output_folder_relative)
    os.makedirs(output_folder, exist_ok=True)

    combined_manifests = []
    
    for request_type, request_properties in task_dict.items():
        n_edits = request_properties.get("count", 0)

        # get n_edits for this request type
        find_dict = {"request_type": request_type}

        requests_processed = 0

        all_requests_iterator = dbm.requests.find(find_dict)

        for request in all_requests_iterator:
            if requests_processed >= n_edits:
                break
            request_user_id = request["user"]

            # create request json manifest
            request_manifest = {
                "request_id": request["_id"],
                "task_type": request["request_type"],
                "task_description": "test_str",
                "instruction": {},
                "request_user": request["user"],
            }

            # print(request.keys())

            if "brep_start" in request:
                try:
                    request_manifest["input_3d_model_images"] = []
                    brep_start_images = dbm.get_brep_images(request["brep_start"], views=request_properties["input_views"])


                    for input_3d_model_src in brep_start_images:
                        input_3d_model_src = os.path.join(dbm.root_dir, input_3d_model_src)
                        input_3d_model_dst = os.path.join(output_folder, os.path.basename(input_3d_model_src))
                        input_3d_model_s3 = os.path.join(s3_base_path, os.path.basename(input_3d_model_src))
                        shutil.copy(input_3d_model_src, input_3d_model_dst)
                        request_manifest["input_3d_model_images"].append(input_3d_model_s3)
                except Exception as e:
                    print(f"Error copying input 3D model for request {request['_id']}: {e}")


            video_key = "30_720_audio"
            if video_key in request:
                video_path = os.path.join(dbm.root_dir, request[video_key])
                video_filename = os.path.basename(video_path)
                video_dst_path = os.path.join(output_folder, video_filename)
                shutil.copy(video_path, video_dst_path)
                video_s3_path = os.path.join(s3_base_path, video_filename)
                request_manifest["instruction"]["video"] = video_s3_path

            if "text" in request and request["text"]:
                request_manifest["instruction"]["text"] = request["text"]

            if "modality" in request:
                request_manifest["instruction"]["modality"] = request["modality"]

            edit_iterator = dbm.edits.find({"request": request["_id"]})

            for edit in edit_iterator:
                if ignore_failed_runs and edit.get("failed_run", False):
                    print(f"Skipping failed run for edit {edit['_id']}")
                    continue

                current_manifest = request_manifest.copy()
                edit_user_id = edit["user"]
                edit_user = dbm.users.find_one({"_id": edit_user_id})
                is_human = edit_user.get("is_human", False)

                # get the number of existing ratings for this edit. Also need to check that "private." is in the rating user id, to check that only groundtruth edits are counted.
                ratings = dbm.ratings.find({"edit": edit["_id"]})
                ratings = [r for r in ratings if "private." in r["user"]]
                n_ratings = len(ratings)
                if n_ratings >= n_ratings_threshold:
                    continue

                keep = False
                if edit_user_id in models:
                    keep = keep or True
                if "other human" in models:
                    keep = keep or (is_human and edit_user_id != request_user_id)
                if "gt human" in models:
                    keep = keep or (is_human and edit_user_id == request_user_id)
                if not keep:
                    continue

                current_manifest["edit_id"] = edit["_id"]
                current_manifest["edit_user"] = edit["user"]

                if "brep_end" in edit:
                    # current_manifest["source-ref"] = []
                    current_manifest["output_3d_model_images"] = []
                    brep_end_images = dbm.get_brep_images(edit["brep_end"], views=request_properties["output_views"])
                    for output_3d_model_src in brep_end_images:
                        output_3d_model_src = os.path.join(dbm.root_dir, output_3d_model_src)
                        output_3d_model_dst = os.path.join(output_folder, os.path.basename(output_3d_model_src))
                        output_3d_model_s3 = os.path.join(s3_base_path, os.path.basename(output_3d_model_src))
                        shutil.copy(output_3d_model_src, output_3d_model_dst)
                        current_manifest["output_3d_model_images"].append(output_3d_model_s3)

                    source_ref_image = dbm.get_brep_images(edit["brep_end"], views=[request_properties["source-ref_view"]])
                    if source_ref_image:
                        source_ref_image = source_ref_image[0]
                        source_ref_image_src = os.path.join(dbm.root_dir, source_ref_image)
                        source_ref_image_dst = os.path.join(output_folder, os.path.basename(source_ref_image_src))
                        source_ref_image_s3 = os.path.join(s3_base_path, os.path.basename(source_ref_image_src))
                        shutil.copy(source_ref_image_src, source_ref_image_dst)
                        current_manifest["source-ref"] = source_ref_image_s3
                    else:
                        print(f"Warning: No source-ref image found for request {request['_id']} edit {edit['_id']}")
                        current_manifest["source-ref"] = None

                combined_manifests.append(current_manifest)
            
            requests_processed += 1


    for m in combined_manifests:
        print(json.dumps(m, indent=2))
    print(f"Total manifests exported: {len(combined_manifests)}")

    # save combined manifest to output folder
    manifest_output_path = os.path.join(output_folder, "ground_truth_manifest")
    with open(manifest_output_path + ".jsonl", "w") as f:
        for m in combined_manifests:
            f.write(json.dumps(m) + "\n")
    with open(manifest_output_path + ".json", "w") as f:
        json.dump(combined_manifests, f, indent=2)
    




def main():
    # Parse command-line arguments
    args = parse_args()
    # Load configuration
    config = load_config(args.config)

    dbm = DatabaseManager(config)

    dbm.clean_db_single_edit_per_user_per_request()

    all_views = ["toprightiso", "front", "back", "top", "bottom", "left", "right"]

    task_dict = {
        "edit": {"count": 99999, "input_views": all_views, "output_views": all_views, "source-ref_view": "toprightiso"},
    }

    models=[
        "gt human",
        "other human",
        "gemini-3-pro_cadquery-script",
        "gpt-5.2_cadquery-script",
        "claude-sonnet-4.5_cadquery-script",
    ]

    export_files(
        config,
        dbm,
        output_folder_relative="export/ground_truth/",
        s3_base_path="s3://bucket-name/dataset-folder/",
        task_dict=task_dict,
        models=models,
        n_ratings_threshold=999,
        # ignore_failed_runs=False,
    )

 

if __name__ == "__main__":
    main()

