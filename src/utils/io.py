import os

def get_brep_path_from_folder(folder_path, extension=".obj"):
    """
    Gets the BREP path from the given folder path.

    Args:
        folder_path (str): The folder path.

    Returns:
        str: The BREP path.
    """
    
    # get all cad files in the folder
    brep_files = [f for f in os.listdir(folder_path) if f.endswith(extension)]
    if len(brep_files) == 0:
        raise ValueError(f"No {extension} files found in the folder: {folder_path}")
    if len(brep_files) > 1:
        raise ValueError(f"Multiple {extension} files found in the folder: {folder_path}. Please provide a single file.")
    brep_path = os.path.join(folder_path, brep_files[0])
    return brep_path
