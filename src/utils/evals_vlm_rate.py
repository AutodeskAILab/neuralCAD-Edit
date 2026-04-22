# from src.scripts.build_instructions_db import load_all_instructions  # Import the instructions_db directly
from src.utils.args import parse_args
from src.utils.process_config import load_config
import os.path as osp

import importlib

from src.utils.db import DatabaseManager



def vlm_rate_eval(config: dict, db: DatabaseManager, edit_id_list) -> None:

    run_folder = osp.join(db.root_dir, db.model_rate_dir)

    for model_key, model_config in config["rating_models"].items():
        if "skip" in model_config and model_config["skip"]:
            print(f"Skipping model {model_key} due to skip flag in config.")
            continue
        
        # use getattr to dynamically load the model class
        model_name = model_config["family"]
        module_path = f"src.vlms.{model_name}"
        module = importlib.import_module(module_path)
        model = module.VLM(model_config, cache=True)

        # insert the model into the database
        db.insert_user(user_id=model_key, email=None, vlm_config=model_config, is_human=False)

        for edit_id in edit_id_list:

            edit = db.edits.find_one({"_id": edit_id})

            request = db.requests.find_one({"_id": edit["request"]})

            if request is None:
                print(f"Request {edit['request']} not found for edit {edit['_id']}. Skipping.")
                continue

            # default request type is "edit"
            if "request_type" not in request:
                request["request_type"] = "edit"

            if request["request_type"] not in model_config["request_types"]:
                continue

            out_fn_base = f"{model_key}_{edit["_id"]}.json"
            output_path = osp.join(run_folder, out_fn_base)

            # compute ratings

            rating_func = getattr(model, model.config["rating_function"])

            response = rating_func(db, edit["_id"], output_path)

            if response is None:
                print(f"Error processing edit {edit['_id']}")
                continue

            id = db.insert_rating(
                user=model_key,
                edit=edit["_id"],
                score_instr=response["score-instruction-understanding"],
                score_quality=response["score-quality"],
                )
            if not id:
                id = db.ratings.find_one({"edit": edit["_id"], "user": model_key})["_id"]
            db.ratings.update_one(
                {"_id": id},
                {"$set": {
                    "score_instr": response["score-instruction-understanding"],
                    "score_quality": response["score-quality"]
                }}
            )


def main():
    # Parse command-line arguments
    args = parse_args()
    # Load configuration
    config = load_config(args.config)

    db = DatabaseManager(config)
    db.print_db_summary()

    all_edits_iterator = db.edits.find()
    edit_id_list = [edit["_id"] for edit in all_edits_iterator]

    vlm_rate_eval(config, db, edit_id_list)


if __name__ == "__main__":
    main()