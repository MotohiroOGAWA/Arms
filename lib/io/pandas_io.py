import pandas as pd
from tqdm import tqdm
from math import ceil
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Literal, List

# Disable RDKit logging
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)  # Only show critical errors, suppress warnings and other messages


def count_lines(file_path: str) -> int:
    result = subprocess.run(['wc', '-l', file_path], stdout=subprocess.PIPE, text=True)
    return int(result.stdout.strip().split()[0])

def read_file_in_chunks(file_path, sep, header, column_names, chunk_size):
    """
    Reads a large file in chunks and yields them one by one.

    Parameters:
        file_path (str): Path to the input file.
        sep (str): Separator used in the file.
        header (int or None): Row number(s) to use as the column names.
        column_names (list): List of column names for the DataFrame.
        chunk_size (int): Number of rows per chunk.

    Yields:
        pd.DataFrame: Each chunk as a DataFrame.
    """
    for chunk in pd.read_csv(
        file_path,
        sep=sep,
        header=header,
        names=column_names,
        chunksize=chunk_size
    ):
        yield chunk


def example_process_chunk(chunk:pd.DataFrame, index, prefix="chunk", save=False, output_dir=None):
    """
    results = run_parallel_processing(
        process_chunk=example_process_chunk,
        process_kwargs={"prefix": "mychunk", "save": True, "output_dir": "chunks"}
    )
    """
    if save and output_dir:
        out_path = os.path.join(output_dir, f"{prefix}_{index:03d}.tsv")
        chunk.to_csv(out_path, sep="\t", index=False)

    return index, chunk.sum(numeric_only=True)

def run_parallel_processing(file_path, sep, header, column_names, chunk_size, num_workers,
                            process_chunk, process_kwargs=None,
                            executor_type: Literal["thread", "process"] = "thread",
                            merging: bool = False,
                            ) -> pd.DataFrame | List[pd.DataFrame]:
    """
    Process file chunks in parallel using ThreadPoolExecutor or ProcessPoolExecutor.

    Parameters:
        file_path (str): Path to the input file.
        sep (str): Delimiter for the file.
        header (int or None): Header row index.
        column_names (list): Column names to use if no header is present.
        chunk_size (int): Number of rows per chunk.
        num_workers (int): Number of threads or processes to use.
        process_chunk (callable): A function to apply to each chunk. Must accept (chunk, index, **kwargs).
        process_kwargs (dict or None): Extra keyword arguments passed to process_chunk.
        executor_type (str): "thread" for ThreadPoolExecutor or "process" for ProcessPoolExecutor.
        merging (bool): If True, merge all results into a single DataFrame.

    Returns:
        list of results (ordered by chunk index)
    """
    if process_kwargs is None:
        process_kwargs = {}

    # Estimate total chunks
    total_lines = count_lines(file_path)
    if header is not None:
        total_lines -= 1  # skip header row
    estimated_chunks = ceil(total_lines / chunk_size)

    # Create chunk iterator
    chunk_iter = read_file_in_chunks(file_path, sep, header, column_names, chunk_size)

    results = []
    futures = []

    # Choose executor type
    ExecutorClass = ThreadPoolExecutor if executor_type == "thread" else ProcessPoolExecutor

    with ExecutorClass(max_workers=num_workers) as executor:
        for i, chunk in enumerate(tqdm(chunk_iter, total=estimated_chunks, desc="Submitting chunks")):
            future = executor.submit(process_chunk, chunk, i, **process_kwargs)
            future.chunk_index = i
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
            result = future.result()
            index = future.chunk_index
            results.append((index, result))


    results.sort(key=lambda x: x[0])
    results = [result for _, result in results]
    
    if merging:
        # Merge all DataFrames into a single DataFrame
        print("Merging all processed chunks into a single DataFrame...")
        merged_df = pd.concat(results, ignore_index=True)
        return merged_df
    else:
        return results
    
def save_parquet(df: pd.DataFrame, file_path: str, preview: bool = True):
    """
    Save DataFrame to a Parquet file.

    Parameters:
        df (pd.DataFrame): DataFrame to save.
        file_path (str): Path to save the Parquet file.
    """
    # Ensure output directory exists and save the final DataFrame
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    df.to_parquet(file_path, index=False)

    if preview:
        # Preview the first 100 rows of the merged DataFrame
        print("Preview of the merged DataFrame:")
        print(df.head(n=5))

        # Save a preview to a TSV file
        preview_path = os.path.splitext(file_path)[0] + "_preview.tsv"
        df.head(n=100).to_csv(preview_path, sep="\t", index=False)


