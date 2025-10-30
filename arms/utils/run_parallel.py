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
    commands: List[str]
) -> None:
    """
    Run a single task in a subprocess.
    Args:
        commands (List[str]): Full command list to execute.
    """
    subprocess.run(
        commands, 
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        )


def run_parallel_subprocesses(
    commands_list: List[List[str]],
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
        futures = [executor.submit(run_in_subprocess, commands) for commands in commands_list]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Parallel tasks"):
            f.result()
