import open3d as o3d
import numpy as np
from PIL import Image

def cad_file_2_o3d_mesh(file_path: str) -> o3d.geometry.TriangleMesh:
    """
    Loads a cad file into an Open3D TriangleMesh.

    Args:
        cad_file_path (str): Path to the cad file.

    Returns:
        o3d.geometry.TriangleMesh: The loaded mesh.
    """

    if file_path.endswith('.obj'):
        mesh = o3d.io.read_triangle_mesh(file_path)
        if not mesh:
            raise ValueError(f"Failed to load cad file: {file_path}")
        return mesh
    
    else:
        raise ValueError(f"Unsupported file format: {file_path}. Only .obj files are supported.")


def render_mesh_to_image(mesh: o3d.geometry.TriangleMesh, output_image_path: str) -> None:
    """
    Renders a mesh to an image using Open3D with improved lighting and edge visibility.

    Args:
        mesh (o3d.geometry.TriangleMesh): The mesh to render.
        output_image_path (str): Path to save the rendered image.
    """
    # Create a visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(mesh)

    # Add lighting
    opt = vis.get_render_option()
    opt.light_on = True
    opt.background_color = np.array([1, 1, 1])  # Set background to white
    opt.mesh_show_back_face = True
    opt.line_width = 1.0  # Thin black lines for edges
    opt.mesh_show_wireframe = True

    # Render the scene
    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()

    # Save the rendered image
    image = (np.asarray(image) * 255).astype(np.uint8)
    Image.fromarray(image).save(output_image_path)

def render_cad_file_to_image(cad_file_path: str, output_image_path: str) -> None:
    """
    Loads a cad file and renders it to an image.

    Args:
        cad_file_path (str): Path to the cad file.
        output_image_path (str): Path to save the rendered image.
    """
    mesh = cad_file_2_o3d_mesh(cad_file_path)
    render_mesh_to_image(mesh, output_image_path)

def align(reference_mesh: o3d.geometry.TriangleMesh, aligning_mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """
    Aligns the aligning mesh to the reference mesh using ICP.

    Args:
        reference_mesh (o3d.geometry.TriangleMesh): The reference mesh.
        aligning_mesh (o3d.geometry.TriangleMesh): The mesh to align.

    Returns:
        o3d.geometry.TriangleMesh: The aligned mesh.
    """
    # Compute point clouds from meshes
    reference_pcd = reference_mesh.sample_points_uniformly(number_of_points=1000)
    aligning_pcd = aligning_mesh.sample_points_uniformly(number_of_points=1000)

    # Perform ICP alignment
    threshold = 0.02  # Distance threshold for ICP
    transformation = o3d.pipelines.registration.registration_icp(
        aligning_pcd, reference_pcd, threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    ).transformation

    # Apply the transformation to the aligning mesh
    aligning_mesh.transform(transformation)
    return aligning_mesh

def compute_IoU(reference_mesh: o3d.geometry.TriangleMesh, aligning_mesh: o3d.geometry.TriangleMesh) -> float:
    """
    Computes the Intersection over Union (IoU) between two meshes.

    Args:
        reference_mesh (o3d.geometry.TriangleMesh): The reference mesh.
        aligning_mesh (o3d.geometry.TriangleMesh): The aligned mesh.

    Returns:
        float: The IoU value.
    """
    # Compute voxel grids
    reference_voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(reference_mesh, voxel_size=0.01)
    aligning_voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(aligning_mesh, voxel_size=0.01)

    reference_voxels = set(tuple(voxel.grid_index) for voxel in reference_voxel_grid.get_voxels())
    aligning_voxels = set(tuple(voxel.grid_index) for voxel in aligning_voxel_grid.get_voxels())

    intersection = len(reference_voxels & aligning_voxels)
    union = len(reference_voxels | aligning_voxels)
    return intersection / union if union > 0 else 0.0

def load_align_compute_IoU(reference_cad_file_path: str, to_align_cad_file_path: str) -> float:
    """
    Aligns two meshes, computes IoU, and renders the result to an image.

    Args:
        reference_mesh (o3d.geometry.TriangleMesh): The reference mesh.
        aligning_mesh (o3d.geometry.TriangleMesh): The mesh to align.
        output_image_path (str): Path to save the rendered image.
    """

    # Load the meshes
    reference_mesh = cad_file_2_o3d_mesh(reference_cad_file_path)

    try:
        to_align_mesh = cad_file_2_o3d_mesh(to_align_cad_file_path)
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return np.inf

    iou = compute_IoU(reference_mesh, to_align_mesh)
    return iou
