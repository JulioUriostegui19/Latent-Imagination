import os  # Operating system interface for directory operations


def ensure_dir(path: str):
    """Create directory if it doesn't exist, with no error if it already exists."""
    # Create directory (and any parent directories) if they don't exist
    # exist_ok=True prevents error if directory already exists
    os.makedirs(path, exist_ok=True)
