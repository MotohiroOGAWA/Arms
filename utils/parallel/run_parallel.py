import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Union, Generator, Any
from tqdm import tqdm


def generate_task_arguments(
    input_files: List[str],
    output_files: List[str],
    input_flag: str = "-i",
    output_flag: str = "-o"
) -> Generator[List[str], None, None]:
    """
    Generate arguments for subprocess tasks.

    Args:
        input_files (List[str]): Input file paths.
        output_files (List[str]): Output file paths.
        input_flag (str): Command-line flag for input.
        output_flag (str): Command-line flag for output.

    Yields:
        List[str]: Argument list for one subprocess.
    """
    for in_file, out_file in zip(input_files, output_files):
        yield [input_flag, in_file, output_flag, out_file]

def _to_module_name(path: str) -> str:
    """Convert file path to module name if needed."""
    if path.endswith(".py"):
        path = os.path.splitext(path)[0] 
        path = path.replace(os.sep, ".") 
    return path

def run_in_subprocess(
    script_path: str,
    args: Union[List[str], Dict[str, Any]]
) -> None:
    """
    Run a single task in a subprocess.

    Args:
        script_path (str): Path to Python script to execute.
        args (list|dict): Arguments for the script.
    """
    module = _to_module_name(script_path)
    command = ["python", "-m", module]

    if isinstance(args, list):
        command.extend(args)
    elif isinstance(args, dict):
        for key, value in args.items():
            if isinstance(value, bool):
                if value:
                    command.append(str(key))
            else:
                command.extend([str(key), str(value)])
    else:
        raise TypeError("args must be list or dict")

    subprocess.run(command, check=True)


def run_parallel_subprocesses(
    script_path: str,
    task_args: List[Union[List[str], Dict[str, Any]]],
    max_workers: int = 4
) -> None:
    """
    Run multiple subprocesses in parallel.

    Args:
        script_path (str): Path to Python script to execute.
        task_args (List): List of argument sets for each subprocess.
        max_workers (int): Number of parallel workers.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_in_subprocess, script_path, args) for args in task_args]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Parallel tasks"):
            f.result()
