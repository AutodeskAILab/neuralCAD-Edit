from src.utils.args import parse_args
from src.utils.process_config import load_config
import os.path as osp
from src.utils.db import DatabaseManager
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import open3d as o3d
from scipy.spatial.distance import cdist
from probreg import cpd
import copy
import multiprocessing


def load_stl_as_point_cloud(stl_path, num_samples=10000):
    """
    Load an STL file and convert it to a point cloud.
    
    Args:
        stl_path (str): Path to the STL file
        num_samples (int): Number of points to sample from the mesh surface
        
    Returns:
        numpy.ndarray: Point cloud as (N, 3) array
    """
    try:
        # Load the mesh
        mesh = o3d.io.read_triangle_mesh(str(stl_path))
        
        if len(mesh.vertices) == 0:
            raise ValueError(f"Failed to load mesh from {stl_path}")
        
        # print(f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
        
        # Sample points uniformly from the mesh surface
        if len(mesh.triangles) > 0:
            point_cloud = mesh.sample_points_uniformly(number_of_points=num_samples)
            points = np.asarray(point_cloud.points)
        else:
            # If no triangles, use vertices directly
            points = np.asarray(mesh.vertices)
            if len(points) > num_samples:
                # Randomly sample if too many points
                indices = np.random.choice(len(points), num_samples, replace=False)
                points = points[indices]
        
        # print(f"Generated point cloud with {len(points)} points")
        return points
        
    except Exception as e:
        raise RuntimeError(f"Error loading STL file {stl_path}: {str(e)}")





def compute_chamfer_distance(points1, points2, normalize=False):
    """
    Compute chamfer distance between two point clouds.
    
    Chamfer distance = mean(min_dist(p1 -> p2)) + mean(min_dist(p2 -> p1))
    
    Args:
        points1 (numpy.ndarray): First point cloud (N1, 3)
        points2 (numpy.ndarray): Second point cloud (N2, 3)
        normalize (bool): Whether to normalize by the diagonal of bounding box
        
    Returns:
        tuple: (chamfer_distance, forward_distance, backward_distance)
    """
    # Compute pairwise distances
    distances_1_to_2 = cdist(points1, points2, metric='euclidean')
    distances_2_to_1 = cdist(points2, points1, metric='euclidean')
    
    # Find minimum distances
    min_distances_1_to_2 = np.min(distances_1_to_2, axis=1)
    min_distances_2_to_1 = np.min(distances_2_to_1, axis=1)
    
    # Compute forward and backward distances
    forward_distance = np.mean(min_distances_1_to_2)
    backward_distance = np.mean(min_distances_2_to_1)
    
    # Chamfer distance is the sum of both directions
    chamfer_dist = forward_distance + backward_distance
    
    if normalize:
        # Normalize by the diagonal of the combined bounding box
        all_points = np.vstack([points1, points2])
        bbox_min = np.min(all_points, axis=0)
        bbox_max = np.max(all_points, axis=0)
        bbox_diagonal = np.linalg.norm(bbox_max - bbox_min)
        
        chamfer_dist /= bbox_diagonal
        forward_distance /= bbox_diagonal
        backward_distance /= bbox_diagonal
    
    return chamfer_dist, forward_distance, backward_distance


def pair_cosine_similarity(f1, f2, db=None):
    return cosine_similarity(
        np.array(f1).reshape(1, -1),
        np.array(f2).reshape(1, -1)
    )[0][0]


def align_point_clouds(source_pc, target_pc, num_points=1000, num_initializations=8):
    initializations = [(0,0,0)]
    initializations.extend([np.random.uniform(0, 2*np.pi, size=3) for _ in range(num_initializations-1)])

    if isinstance(source_pc, np.ndarray):
        source_pc_o3d = o3d.geometry.PointCloud()
        source_pc_o3d.points = o3d.utility.Vector3dVector(source_pc)
    else:
        source_pc_o3d = copy.deepcopy(source_pc)

    if isinstance(target_pc, np.ndarray):
        target_pc_o3d = o3d.geometry.PointCloud()
        target_pc_o3d.points = o3d.utility.Vector3dVector(target_pc)
    else:
        target_pc_o3d = copy.deepcopy(target_pc)

    # downsample
    source_pc_ds = source_pc_o3d.random_down_sample(min(1.0, float(num_points) / len(source_pc_o3d.points)))
    target_pc_ds = target_pc_o3d.random_down_sample(min(1.0, float(num_points) / len(target_pc_o3d.points)))

    best_sigma2 = float('inf')
    best_tf_param = None
    best_R = None
    best_center = None

    for cur_initialization in initializations:
        # Random rotation about bbox center
        R = o3d.geometry.get_rotation_matrix_from_xyz(cur_initialization)
        bbox_center = source_pc_ds.get_axis_aligned_bounding_box().get_center()
        source_pc_rotated = copy.deepcopy(source_pc_ds).rotate(R, center=bbox_center)

        registration_result = cpd.registration_cpd(source_pc_rotated, target_pc_ds, tol=1e-10, maxiter=100)

        print(registration_result, registration_result.sigma2)
        tf_param, _, _ = registration_result
        # o3d.visualization.draw_geometries([target_pc, source_pc_rotated])

        if registration_result.sigma2 < best_sigma2:
            best_sigma2 = registration_result.sigma2
            best_tf_param = tf_param
            best_R = R
            best_center = bbox_center

    # Apply the best transformation to the original source point cloud
    # First rotate, then apply CPD transformation
    source_pc_o3d_rotated = copy.deepcopy(source_pc_o3d)
    source_pc_o3d_rotated.rotate(best_R, center=best_center)
    
    # Apply CPD transformation and convert back to numpy array
    transformed_points = best_tf_param.transform(np.asarray(source_pc_o3d_rotated.points))
    
    return transformed_points


def chamfer_similarity(source_brep, target_brep, db, pre_align=False):
    """
    Computes the Chamfer distance between two BREP objects.
    
    Args:
        source_brep (str): Path to the source BREP file.
        target_brep (str): Path to the target BREP file.
        db (DatabaseManager): Database manager instance for logging or retrieval.
        
    Returns:
        float: 1 / Chamfer distance between the two BREP objects.
    """

    # allows for lists of stls etc.
    if isinstance(source_brep, list):
        source_brep = source_brep[0]
    if isinstance(target_brep, list):
        target_brep = target_brep[0]

    pts1 = load_stl_as_point_cloud(osp.join(db.root_dir, source_brep))
    pts2 = load_stl_as_point_cloud(osp.join(db.root_dir,target_brep))

    if pts1.shape[0] == 0 or pts2.shape[0] == 0:
        print(f"Skipping Chamfer similarity for {source_brep} and {target_brep} due to empty point clouds.")
        return 100.0

    if pre_align:
        pts2 = align_point_clouds(pts2, pts1)

    chamfer_dist = compute_chamfer_distance(pts1, pts2, normalize=False)[0]

    return 1.0 / (chamfer_dist + 1e-8)  # Avoid division by zero




def align_meshes(mesh_source, mesh_target, num_points=1000, num_initializations=8):
    """
    Returns the transformed mesh_source aligned to mesh_target.
    """

    # convert meshes to point clouds
    source_pc = mesh_source.sample_points_uniformly(number_of_points=num_points)
    target_pc = mesh_target.sample_points_uniformly(number_of_points=num_points)

    initializations = [(0,0,0)]
    initializations.extend([np.random.uniform(0, 2*np.pi, size=3) for _ in range(num_initializations-1)])

    best_sigma2 = float('inf')
    best_tf_param = None
    best_R = None
    best_center = None

    for cur_initialization in initializations:
        # Random rotation about bbox center
        R = o3d.geometry.get_rotation_matrix_from_xyz(cur_initialization)
        bbox_center = source_pc.get_axis_aligned_bounding_box().get_center()
        source_pc_rotated = copy.deepcopy(source_pc).rotate(R, center=bbox_center)

        # Compute the transformation parameters using CPD
        registration_result = cpd.registration_cpd(source_pc_rotated, target_pc, tol=1e-10, maxiter=100)

        print(registration_result, registration_result.sigma2)
        tf_param, _, _ = registration_result
        # o3d.visualization.draw_geometries([target_pc, source_pc_rotated])

        if registration_result.sigma2 < best_sigma2:
            best_sigma2 = registration_result.sigma2
            best_tf_param = tf_param
            best_R = R
            best_center = bbox_center

    # Apply the best transformation to the source mesh
    transformed_mesh = copy.deepcopy(mesh_source)
    # transformed_vertices = best_tf_param.transform(np.asarray(mesh_source.vertices))

    # rotate transformed_mesh
    transformed_mesh.rotate(best_R, center=best_center)

    transformed_vertices = best_tf_param.transform(np.asarray(transformed_mesh.vertices))
    transformed_mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)
    return transformed_mesh


def _iou_worker(result_queue, source_brep, target_brep, db_root_dir, voxel_divisor, pre_align):
    """Worker function for iou that runs in a separate process."""
    try:
        db_proxy = type('db', (object,), {'root_dir': db_root_dir})()
        result = _iou_impl(source_brep, target_brep, db_proxy, voxel_divisor, pre_align)
        result_queue.put(("ok", result))
    except Exception as e:
        result_queue.put(("error", str(e)))


def _iou_impl(source_brep, target_brep, db, voxel_divisor=100, pre_align=False):
    """
    Actual IoU implementation.
    """
    # allows for lists of stls etc.
    if isinstance(source_brep, list):
        source_brep = source_brep[0]
    if isinstance(target_brep, list):
        target_brep = target_brep[0]

    o3d_source = o3d.io.read_triangle_mesh(osp.join(db.root_dir, source_brep))
    o3d_target = o3d.io.read_triangle_mesh(osp.join(db.root_dir, target_brep))

    if pre_align:
        o3d_target = align_meshes(o3d_target, o3d_source)

    bbox_source = o3d_source.get_axis_aligned_bounding_box()
    bbox_target = o3d_target.get_axis_aligned_bounding_box()
    diagonal_source = np.linalg.norm(bbox_source.get_max_bound() - bbox_source.get_min_bound())
    diagonal_target = np.linalg.norm(bbox_target.get_max_bound() - bbox_target.get_min_bound())
    voxel_size = min(diagonal_source, diagonal_target) / float(voxel_divisor)

    reference_voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(o3d_source, voxel_size=voxel_size)
    aligning_voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(o3d_target, voxel_size=voxel_size)

    reference_voxels = set(tuple(voxel.grid_index) for voxel in reference_voxel_grid.get_voxels())
    aligning_voxels = set(tuple(voxel.grid_index) for voxel in aligning_voxel_grid.get_voxels())

    intersection = len(reference_voxels & aligning_voxels)
    union = len(reference_voxels | aligning_voxels)
    return float(intersection) / float(union) if union > 0 else 0.0


def iou(source_brep, target_brep, db, voxel_divisor=100, pre_align=False, timeout_seconds=120):
    """
    Computes the Intersection over Union (IoU) between two BREP objects.
    Runs in a subprocess with a hard timeout to avoid hangs.
    
    Args:
        source_brep (str): Path to the source BREP file.
        target_brep (str): Path to the target BREP file.
        db (DatabaseManager): Database manager instance.
        voxel_divisor (int): Controls voxel resolution.
        pre_align (bool): Whether to align meshes before computing IoU.
        timeout_seconds (int): Hard timeout in seconds.
        
    Returns:
        float: IoU between the two BREP objects, or 0.0 on timeout/error.
    """
    result_queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_iou_worker,
        args=(result_queue, source_brep, target_brep, db.root_dir, voxel_divisor, pre_align)
    )
    p.start()
    p.join(timeout=timeout_seconds)
    if p.is_alive():
        p.kill()
        p.join()
        print(f"IoU timed out after {timeout_seconds}s for {source_brep} / {target_brep}")
        return 0.0
    if not result_queue.empty():
        status, value = result_queue.get_nowait()
        if status == "ok":
            return value
        else:
            print(f"IoU error for {source_brep} / {target_brep}: {value}")
            return 0.0
    return 0.0
  
# def iou(source_brep, target_brep, db, voxel_divisor=100, pre_align=False):
#     """
#     Computes the Intersection over Union (IoU) between two BREP objects.
    
#     Args:
#         source_brep (str): Path to the source BREP file.
#         target_brep (str): Path to the target BREP file.
#         db (DatabaseManager): Database manager instance for logging or retrieval.
        
#     Returns:
#         float: IoU between the two BREP objects.
#     """
    
#     # allows for lists of stls etc.
#     if isinstance(source_brep, list):
#         source_brep = source_brep[0]
#     if isinstance(target_brep, list):
#         target_brep = target_brep[0]

#     o3d_source = o3d.io.read_triangle_mesh(osp.join(db.root_dir, source_brep))
#     o3d_target = o3d.io.read_triangle_mesh(osp.join(db.root_dir, target_brep))


#     if pre_align:
#         # align the target to the source
#         o3d_target = align_meshes(o3d_target, o3d_source)
#         # o3d.visualization.draw_geometries([o3d_source])
#         # o3d.visualization.draw_geometries([o3d_target])
#         # o3d.visualization.draw_geometries([o3d_target_align])

#     # # make o3d_source red and o3d_target_align blue
#     # o3d_source.paint_uniform_color([1, 0, 0])
#     # o3d_target.paint_uniform_color([0, 0, 1])

#     # vis = o3d.visualization.Visualizer()
#     # vis.create_window()
#     # vis.add_geometry(o3d_source)
#     # vis.add_geometry(o3d_target)

#     # # Get render options and set transparency
#     # render_option = vis.get_render_option()
#     # render_option.mesh_show_back_face = True
    
#     # vis.run()
    

#     bbox_source = o3d_source.get_axis_aligned_bounding_box()
#     bbox_target = o3d_target.get_axis_aligned_bounding_box()
#     diagonal_source = np.linalg.norm(bbox_source.get_max_bound() - bbox_source.get_min_bound())
#     diagonal_target = np.linalg.norm(bbox_target.get_max_bound() - bbox_target.get_min_bound())
#     voxel_size = min(diagonal_source, diagonal_target) / float(voxel_divisor)

#     # Compute voxel grids
#     reference_voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(o3d_source, voxel_size=voxel_size)
#     aligning_voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(o3d_target, voxel_size=voxel_size)

#     # # show voxel grids
#     # vis = o3d.visualization.Visualizer()
#     # vis.create_window()
#     # vis.add_geometry(reference_voxel_grid)
#     # vis.add_geometry(aligning_voxel_grid)
#     # vis.run()

#     reference_voxels = set(tuple(voxel.grid_index) for voxel in reference_voxel_grid.get_voxels())
#     aligning_voxels = set(tuple(voxel.grid_index) for voxel in aligning_voxel_grid.get_voxels())

#     intersection = len(reference_voxels & aligning_voxels)
#     union = len(reference_voxels | aligning_voxels)
#     return float(intersection) / float(union) if union > 0 else 0.0


def run_feature_gt_similarity_eval(config: dict, dbm: DatabaseManager, feature_key: str = "feature_dino", description: str = "dino similarity", distance_func=pair_cosine_similarity, request_type: str = "edit", distance_func_kwargs=None):
    all_requests_iterator = dbm.requests.find({"request_type": request_type})

    for request in all_requests_iterator:

        brep_start_id = request["brep_start"]
        gt_user = request["user"]
        brep_start = dbm.breps.find_one({"_id": brep_start_id})
        start_feature = brep_start.get(feature_key, None)

        gt_edit = dbm.edits.find_one({"request": request["_id"], "user": gt_user})
        gt_brep = dbm.breps.find_one({"_id": gt_edit["brep_end"]})
        gt_feature = gt_brep.get(feature_key, None)

        # get edits with the same request id and different user
        all_other_user_edits = dbm.edits.find({"request": request["_id"], "user": {"$ne": gt_user}})

        all_edits = dbm.edits.find({"request": request["_id"]})
        all_edits = list(all_edits)

        gt_sim_str = f"{description} gt"
        start_sim_str = f"{description} start"

        for edit in all_other_user_edits:
            valid_user = False
            user = dbm.users.find_one({"_id": edit["user"]})


            if user["_id"]  in config["benchmark_eval_users"][request_type]:
                valid_user = True
            if "other human" in config["benchmark_eval_users"][request_type] and user.get("is_human", True):
                valid_user = True

            if not valid_user:
                print(f"Skipping edit {edit['_id']} because user {edit['user']} is not in benchmark_eval_users")
                continue

            brep_end_id = edit["brep_end"]
            brep_end = dbm.breps.find_one({"_id": brep_end_id})

            if brep_end is None:
                print(f"Skipping edit {edit['_id']} because brep_end is None")
                continue

            # get the feature of the end brep
            edit_feature = brep_end.get(feature_key, None)


            if gt_feature is None or edit_feature is None:
                print(f"Skipping edit {edit['_id']} because features are missing")
                continue

            if dbm.rating_exists("similarity_eval", edit["_id"]):
                rating = dbm.ratings.find_one({"edit": edit["_id"], "user": "similarity_eval"})
                rating_id = rating["_id"]
                if gt_sim_str in rating and start_sim_str in rating:
                    print(f"Skipping edit {edit['_id']} because ratings already exist")
                    continue
            else:
                rating_id = dbm.insert_rating(
                    user="similarity_eval",
                    edit=edit["_id"]
                )


            print(f"Computing {description} for edit {edit['_id']} with gt {gt_edit['_id']} and request {request['_id']}")

            gt_similarity = distance_func(gt_feature, edit_feature, db=dbm, **(distance_func_kwargs or {}))
            start_similarity = distance_func(start_feature, edit_feature, db=dbm, **(distance_func_kwargs or {})) if start_feature and edit_feature else 0.0

            dbm.ratings.update_one(
                {"_id": rating_id},
                {"$set": {
                    gt_sim_str: gt_similarity,
                    start_sim_str: start_similarity
                }}
            )


def run_clip_similarity_eval(config: dict, dbm: DatabaseManager, text_feature_key: str = "feature_clip_text", vis_feature_key: str = "feature_clip_visual", description: str = "clip similarity", distance_func=pair_cosine_similarity, request_type: str = "text2brep"):
    all_requests_iterator = dbm.requests.find({"request_type": request_type})
    for request in all_requests_iterator:

        gt_user = request["user"]
        text_feature = request.get(text_feature_key, None)

        # get edits with the same request id and different user
        all_other_user_edits = dbm.edits.find({"request": request["_id"], "user": {"$ne": gt_user}})

        for edit in all_other_user_edits:
            valid_user = False
            user = dbm.users.find_one({"_id": edit["user"]})
            if user["_id"]  in config["benchmark_eval_users"][request_type]:
                valid_user = True
            if "other human" in config["benchmark_eval_users"][request_type] and user.get("is_human", True):
                valid_user = True

            if not valid_user:
                print(f"Skipping edit {edit['_id']} because user {edit['user']} is not in benchmark_eval_users")
                continue

            brep_end_id = edit["brep_end"]
            brep_end = dbm.breps.find_one({"_id": brep_end_id})

            if brep_end is None:
                print(f"Skipping edit {edit['_id']} because brep_end is None")
                continue

            # get the features of the end brep
            edit_vis_feature = brep_end.get(vis_feature_key, None)

            cos_sim = pair_cosine_similarity(text_feature, edit_vis_feature, db=dbm)

            dbm.insert_rating(
                user="similarity_eval",
                edit=edit["_id"]
            )
            rating_id = dbm.ratings.find_one({"edit": edit["_id"], "user": "similarity_eval"})["_id"]
            dbm.ratings.update_one(
                {"_id": rating_id},
                {"$set": {
                    description: cos_sim
                }}
            )



def main():
    # Parse command-line arguments
    args = parse_args()
    # Load configuration
    config = load_config(args.config)

    dbm = DatabaseManager(config)

    run_feature_gt_similarity_eval(config, dbm, feature_key="feature_dino", description="dino similarity", distance_func=pair_cosine_similarity, request_type="edit")

    run_feature_gt_similarity_eval(config, dbm, feature_key="stl", description="chamfer similarity", distance_func=chamfer_similarity, request_type="edit")

    run_clip_similarity_eval(config, dbm, text_feature_key="feature_clip_text", vis_feature_key="feature_clip_visual", description="clip similarity", distance_func=pair_cosine_similarity, request_type="text2brep")

if __name__ == "__main__":

    # create dummy db object with just a root_dir field
    dmb = type('db', (object,), {})()

    dmb.root_dir = "/Users/perrett/fusion_data/edit_database/vtest/"
    gt = "breps/sketch2brep_requester_4512737050977123573.stl"
    human = "breps/dummy_5512737050977123574.stl"
    print(iou(gt, human, dmb, pre_align=True, voxel_divisor=100))

    # dmb.root_dir = "/Users/perrett/fusion_data/edit_database/v1/"
    # gt = "breps/3YH2WFSRM22W7DKT_1752836805.728848.stl"
    # human = "breps/3YH2WFSRM22W7DKT_1752836805.728848.stl"
    # print(chamfer_similarity(gt, human, dmb, pre_align=False))
    # print(chamfer_similarity(gt, human, dmb, pre_align=True))

    # main()