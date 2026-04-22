import json

def load_config(config_path: str) -> dict:
    """
    Loads a configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Parsed configuration.
    """
    with open(config_path, "r") as config_file:
        return json.load(config_file)
