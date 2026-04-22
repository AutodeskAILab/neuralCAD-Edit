import argparse

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Build instructions database.")
    parser.add_argument("--config", required=True, help="Path to the configuration file.")
    parser.add_argument(
        "--model-keys",
        nargs="*",
        default=None,
        help="Specify the model keys to be run (space-separated). If not provided, all models will be run.",
    )
    return parser.parse_args()
