from src.utils.args import parse_args
from src.utils.process_config import load_config
import os
import os.path as osp
import json
from src.utils.db import DatabaseManager
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def parse_rating(rating: dict) -> dict:
    """
    Parses a rating dictionary to extract relevant information.
    
    Args:
        rating (dict): The rating dictionary to parse.
        
    Returns:
        dict: A dictionary of metric: score containing the parsed information.
    """

    if all(k in rating["user"] for k in ["gemini", "rating"]):
    # if "gemini" in rating["user"] and "rating" in rating["user"]:
        user = rating["user"]
        return {
            f"{user}_instruction": (rating["score_instr"] - 1.0) / 6.0,
            f"{user}_quality": (rating["score_quality"] - 1.0) / 6.0,
        }
    
    if rating["user"] == "similarity_eval":
        return_dict = {}
        if "dino similarity gt" in rating:
            return_dict["dino-v2_similarity"] = rating["dino similarity gt"]
        if "chamfer similarity gt" in rating:
            return_dict["chamfer_similarity"] = rating["chamfer similarity gt"]
        if "clip similarity" in rating:
            return_dict["clip_similarity"] = rating["clip similarity"]
            return_dict["clip_threshold_pass"] = 1 if rating["clip similarity"] > 0.242 else 0
        if "iou gt" in rating:
            return_dict["iou"] = rating["iou gt"]
        return return_dict
    
    if "private." in rating["user"]:
        user = rating["user"]
        print("private user rating found:", rating)

        return {
            f"{user}_instruction": ((rating["score_instr"] - 1.0) / 6.0 if "score_instr" in rating and rating["score_instr"] is not None else None),
            f"{user}_quality": ((rating["score_quality"] - 1.0) / 6.0 if "score_quality" in rating and rating["score_quality"] is not None else None),
        }
    
    return None

def plot_ratings(config, scores: dict, difficulty: str = "all", request_type: str = "edit", mode="ratings"):

    fig, ax = plt.subplots(figsize=(12, 6))

    out_dir = osp.join(config["storage_dir"]["path"], "results")
    os.makedirs(out_dir, exist_ok=True)
    out_fn = osp.join(out_dir, f"{mode}_{request_type}_{difficulty}.png")

    
    # Prepare data for grouped bar chart
    # users = list(scores.keys())
    users = config["benchmark_eval_users"][request_type]
    metrics = set()
    for user_scores in scores.values():
        metrics.update(user_scores.keys())
    metrics = list(metrics)
    metrics.sort()  # Sort metrics for consistent ordering

    print(metrics)


    # Position bars - group by user first, then by metric
    bar_width = 0.8 / len(metrics)
    x = np.arange(len(users))

    print(scores)

    for i, metric in enumerate(metrics):
        metric_means = []
        for user in users:
            values = scores.get(user, {}).get(metric, {})

            values = [v if v is not None else 0.0 for v in values.values()]

            mean_score = np.mean(values) if values else 0.0
            metric_means.append(mean_score)
            # print(f"{user} - {metric}: {mean_score:.4f} (n={len(values)})")
        
        pos = x + i * bar_width - (len(metrics) - 1) * bar_width / 2
        ax.bar(pos, metric_means, width=bar_width, label=metric)
    
    ax.set_ylabel("Mean Score")
    ax.set_xlabel("Model")
    ax.set_title(f"Task: {request_type} {mode}, Difficulty: {difficulty}.")
    ax.set_xticks(x)
    ax.set_xticklabels(users)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_fn)
    # plt.show()

def display_rating_results(config: dict, dbm: DatabaseManager, difficulty: str = "all", request_fields={"eval_vis_multi": True, "eval_geometric": True}, request_type="edit"):

    request_fields["request_type"] = request_type

    if difficulty == "all":
        pass
    else:
        request_fields["difficulty"] = difficulty
        
    request_ids = dbm.requests.find(request_fields)
    request_ids = [request["_id"] for request in request_ids]
    request_ids.sort()


    # scores = {u: [] for u in config["benchmark_eval_users"]}
    scores = {}

    # loop over all ratings in the database
    ratings_iterator = dbm.ratings.find()
    for rating in ratings_iterator:

        edit = dbm.edits.find_one({"_id": rating["edit"]})

        if not edit:
            # print(f"rating {rating['_id']} has no associated edit , skipping.")
            # print(rating)
            continue

        if "request" not in edit:
            # print(f"Edit {edit['_id']} has no request field, skipping.")
            # print(edit)
            continue
        edit_request_id = edit["request"]
        request = dbm.requests.find_one({"_id": edit_request_id})

        if not request:
            # print(f"Edit {edit['_id']} has no associated request , skipping.")
            # print(edit)
            continue

        user = dbm.users.find_one({"_id": edit["user"]})

        if edit_request_id not in request_ids:
            continue

        if request["request_type"] != request_type:
            continue

        valid_user = False
        if user["_id"] in config["benchmark_eval_users"][request_type]:
            valid_user = True
            valid_user_id = user["_id"]
        if "other human" in config["benchmark_eval_users"][request_type] and user.get("is_human", True) and edit["user"] != request["user"]:
            valid_user = True
            valid_user_id = "other human"
        if "gt human" in config["benchmark_eval_users"][request_type] and edit["user"] == request["user"]:
            valid_user = True
            valid_user_id = "gt human"



        if not valid_user:
            continue

        print(rating)


        metrics = parse_rating(rating)

        if not metrics:
            continue

        for k, v in metrics.items():
            user_scores = scores.get(valid_user_id, {})
            scores_dict = user_scores.get(k, {})
            scores_dict[edit_request_id] = v
            # scores_list.append(v)
            user_scores[k] = scores_dict
            scores[valid_user_id] = user_scores

    all_metrics = set()
    for user_scores in scores.values():
        all_metrics.update(user_scores.keys())

    print(all_metrics)

    # for every request_id without a score in scores, add a placeholder
    for request_id in request_ids:
        for user_id in config["benchmark_eval_users"][request_type]:
            if user_id not in scores:
                print(f"User {user_id} not in scores, adding placeholder.")
                scores[user_id] = {}
            for metric in all_metrics:
                if metric not in scores.get(user_id, {}):
                    scores[user_id][metric] = {}
                if request_id not in scores.get(user_id, {}).get(metric, {}):
                    scores[user_id][metric][request_id] = None

    plot_ratings(config, scores, difficulty=difficulty, request_type=request_type, mode="ratings")

    return scores




def display_ranking_results(config: dict, dbm: DatabaseManager, difficulty: str = "all", request_fields={"eval_vis_multi": True, "eval_geometric": True}, request_type="edit"):
    if difficulty == "all":
        pass
    else:
        request_fields["difficulty"] = difficulty
    request_ids = dbm.requests.find(request_fields)
    request_ids = [request["_id"] for request in request_ids]

    # scores = {u: [] for u in config["benchmark_eval_users"]}
    scores = {}

    # loop over all rankings in the database
    rankings_iterator = dbm.rankings.find()
    for ranking in rankings_iterator:

        if ranking["request"] not in request_ids:
            continue

        # print(ranking)

        request_id = ranking["request"]
        request_user = dbm.requests.find_one({"_id": request_id})["user"]
        request = dbm.requests.find_one({"_id": request_id})

        if request["request_type"] != request_type:
            continue

        valid_rankings = []
        found_human = False
        for edit_id in ranking["ranked_edits"]:
            for edit in dbm.edits.find({"_id": edit_id}):

                if edit["user"] == request_user:
                    if "gt human" in config["benchmark_eval_users"][request_type]:
                        valid_user_id = "gt human"
                    else:
                        continue
                elif dbm.users.find_one({"_id": edit["user"]}).get("is_human", True):
                    if found_human:
                        continue
                    else:
                        found_human = True
                        valid_user_id = "other human"
                else:
                    if edit["user"] in config["benchmark_eval_users"][request_type]:
                        valid_user_id = edit["user"]
                    else:
                        continue

                valid_rankings.append(valid_user_id)

        if len(valid_rankings) != len(config["benchmark_eval_users"][request_type]):
            print(f"Skipping request {request_id} as it does not have all required users.")
            print(valid_rankings)
            continue

        for i, valid_user in enumerate(valid_rankings):
            user_scores = scores.get(valid_user, {})
            scores_list = user_scores.get(ranking["user"], [])
            scores_list.append(i)
            user_scores[ranking["user"]] = scores_list
            scores[valid_user] = user_scores

    plot_ratings(config, scores, difficulty=difficulty, request_type=request_type, mode="rankings")

    return scores



def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding Axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def all_tasks_radar_plot(config: dict, dbm: DatabaseManager, results: dict, chosen_models=None, primary_result_keys_override=None, save=True):
    """
    Create a radar plot showing performance across different benchmark types.
    
    Args:
        config: Configuration dictionary with primary_result_keys
        dbm: DatabaseManager instance
        results: Dictionary organized as benchmark_type -> users -> metric -> scores_dict. scores_dict is a dictionary mapping edit_request_id to score.

    Returns:
        tuple: (fig, ax) matplotlib figure and axes objects, or (None, None) if plot cannot be created
    """

    if chosen_models:
        filtered_results = {}
        for task in results:
            filtered_results[task] = {model: data for model, data in results[task].items() if model in chosen_models}
        results = filtered_results

    # Select just the primary result keys
    if primary_result_keys_override:
        primary_result_keys = primary_result_keys_override
    else:
        primary_result_keys = config.get("primary_result_keys", None)
    if primary_result_keys is None:
        print("No primary result keys specified in config, using all keys.")

    # Get benchmark types and collect all users across all benchmarks
    benchmark_types = list(results.keys())
    if len(benchmark_types) < 3:
        print(f"Need at least 3 benchmark types for radar plot, only have {len(benchmark_types)}")
        return None, None
    
    all_users = set()
    for benchmark_data in results.values():
        all_users.update(benchmark_data.keys())
    all_users = sorted(list(all_users))
    
    if len(all_users) == 0:
        print("No users found in results data")
        return None, None

    # Set up radar chart
    N = len(benchmark_types)
    theta = radar_factory(N, frame='polygon')
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='radar'))
    
    # Colors for different users
    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k', 'orange', 'purple', 'brown']
    
    # Prepare data for each user
    for i, user in enumerate(all_users):
        user_scores = []
        
        for benchmark_type in benchmark_types:
            if benchmark_type in results and user in results[benchmark_type]:
                # Get the primary metric for this benchmark type
                primary_metric = primary_result_keys.get(benchmark_type)
                user_data = results[benchmark_type][user]

                if not primary_metric or primary_metric not in user_data:
                    print(f"No primary metric specified for benchmark type '{benchmark_type}', using first available metric.")
                    primary_metric = list(user_data.keys())[0] if user_data else None
                
                # Calculate mean score for this metric
                scores = user_data[primary_metric]
                if isinstance(scores, list) and len(scores) > 0:
                    mean_score = np.mean(scores)
                elif isinstance(scores, (int, float)):
                    mean_score = scores
                elif isinstance(scores, dict) and len(scores) > 0:
                    mean_score = np.mean([v if v is not None else 0.0 for v in list(scores.values())])
                else:
                    mean_score = 0.0

            else:
                mean_score = 0.0
                print(f"Warning: No data for user '{user}' in benchmark '{benchmark_type}'")
            
            user_scores.append(mean_score)
        
        # Plot the user's scores
        color = colors[i % len(colors)]
        ax.plot(theta, user_scores, 'o-', linewidth=2, color=color, label=user, markersize=6)
        ax.fill(theta, user_scores, facecolor=color, alpha=0.15)
        
        print(f"User '{user}' scores: {[f'{score:.3f}' for score in user_scores]}")
    
    # Set up the plot
    ax.set_varlabels(benchmark_types)
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0, 1.35)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()

    if save:
        out_dir = osp.join(config["storage_dir"]["path"], "results")
        os.makedirs(out_dir, exist_ok=True)
        fig_fn = osp.join(out_dir, "radar_plot_all_tasks.png")
        plt.savefig(fig_fn, dpi=300, bbox_inches='tight')
    
    return fig, ax





def main():
    # Parse command-line arguments
    args = parse_args()
    # Load configuration
    config = load_config(args.config)

    dbm = DatabaseManager(config)

    dbm.print_db_summary()

    display_rating_results(config, dbm, difficulty="all", request_type="edit")
  

    out_dir = osp.join(config["storage_dir"]["path"], "results")
    result_path = osp.join(out_dir, "all_results.json")
    with open(result_path, "r") as f:
        results = json.load(f)

    all_tasks_radar_plot(config=config, dbm=None, results=results)



if __name__ == "__main__":
    main()