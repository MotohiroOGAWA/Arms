import os

def derive_file_path(
    input_file: str,
    suffix: str = "",
    ext: str | None = None,
    ensure_unique: bool = True
) -> str:
    """
    Derive a new file path based on the given input file.

    Args:
        input_file (str): Base file path used to derive the new path.
        suffix (str): String to append before the extension (default: "").
        ext (str, optional): Extension for the new file. If None, reuse the input file's extension.
        ensure_unique (bool): If True, avoid overwriting by appending _1, _2, ... if needed.

    Returns:
        str: The derived file path.
    """
    dir_path = os.path.dirname(os.path.abspath(input_file))
    base_name, input_ext = os.path.splitext(os.path.basename(input_file))
    ext = ext or input_ext

    candidate = os.path.join(dir_path, f"{base_name}{suffix}{ext}")

    # if no suffix and path is identical to the input file → add "_0"
    if candidate == os.path.abspath(input_file):
        candidate = os.path.join(dir_path, f"{base_name}_0{ext}")

    # ensure unique name if requested
    if ensure_unique:
        counter = 1
        while os.path.exists(candidate):
            candidate = os.path.join(dir_path, f"{base_name}{suffix}_{counter}{ext}")
            counter += 1

    return candidate
