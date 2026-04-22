from src.utils.args import parse_args
from src.utils.process_config import load_config
import os.path as osp
from src.utils.db import DatabaseManager

from src.utils.evals_vlm_rate import vlm_rate_eval
from src.utils.evals_feature_geometric import run_feature_gt_similarity_eval, pair_cosine_similarity, chamfer_similarity, iou
from src.utils.visualise_results import display_rating_results
from src.utils.extract_features import extract_dino, extract_clip_visual


def run_benchmark_evals(db: DatabaseManager, config: dict, benchmark_type=None) -> None:

    request_find_dict = {"request_type": "edit"}
    if benchmark_type:
        request_find_dict[benchmark_type] = True

    # select requests and edits for the "edit" request type
    all_requests_iterator = db.requests.find(request_find_dict)
    request_id_list = [request["_id"] for request in all_requests_iterator]

    # if there are multiple edits by the same user for the same request, only keep the latest one
    edit_id_list = db.get_latest_edit_ids(request_id_list)

    # extract features from requests and edits
    feature_info = []

    for request_id in request_id_list:
        request = db.requests.find_one({"_id": request_id})
        brep_id = request["brep_start"]
        if not brep_id:
            continue
        frame_path = db.get_brep_images(brep_id)
        frame_path = osp.join(db.root_dir, frame_path[0]) if frame_path else None
        info = {"request_or_edit": "request", "id": request["_id"], "frame_path": frame_path, "brep_id": brep_id}
        feature_info.append(info)

    for edit_id in edit_id_list:
        edit = db.edits.find_one({"_id": edit_id})
        brep_id = edit["brep_end"]
        if not brep_id:
            continue
        frame_path = db.get_brep_images(brep_id)
        frame_path = osp.join(db.root_dir, frame_path[0]) if frame_path else None
        info = {"request_or_edit": "edit", "id": edit["_id"], "frame_path": frame_path, "brep_id": brep_id}
        feature_info.append(info)
    
    # extract features
    extract_dino(config=config, db=db, feature_info=feature_info)
    extract_clip_visual(config=config, db=db, feature_info=feature_info)

    # do feature similarity computations
    run_feature_gt_similarity_eval(config=config, dbm=db, feature_key="feature_dino", description="dino similarity", distance_func=pair_cosine_similarity, request_type="edit")
    run_feature_gt_similarity_eval(config=config, dbm=db, feature_key="stl", description="chamfer similarity", distance_func=chamfer_similarity, request_type="edit")
    run_feature_gt_similarity_eval(config=config, dbm=db, feature_key="stl", description="iou", distance_func=iou, request_type="edit")

    # run vlm evals
    vlm_rate_eval(config=config, db=db, edit_id_list=edit_id_list)
    # vlm_rank_eval(config=config, db=db, request_id_list=request_id_list)

    # plot results
    request_fields = config.get("request_fields", {})

    rating_results = {}
    for difficulty in ["easy", "medium", "hard"]:
        rating_results[f"edit_{difficulty}"] = display_rating_results(config=config, dbm=db, difficulty=difficulty, request_type="edit", request_fields=request_fields)
    ranking_results = {}

    # merge and return results
    results = {**rating_results, **ranking_results}
    return results

def main():
    # Parse command-line arguments
    args = parse_args()
    # Load configuration
    config = load_config(args.config)

    db = DatabaseManager(config)
    db.print_db_summary()

    run_benchmark_evals(db, config)


if __name__ == "__main__":
    main()