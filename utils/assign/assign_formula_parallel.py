import os
import argparse
from tqdm import tqdm
from cores.MassEntity.MassEntityCore.MSDataset import MSDataset
from utils.parallel.run_parallel import *
from utils.parallel.dataset_executor import split_dataset, merge_datasets

SCRIPT_PATH = "utils/assign/assign_formula.py"

# python -m utils.assign.assign_formula_parallel -i data/raw/MoNA/positive/filtered_mona_positive.hdf5 -o data/raw/MoNA/positive/output_assign_mona_positive.hdf5 --mode PossibleFormula --chunk_size 100 --num_workers 1 --mass_tolerance 0.01 --timeout_seconds 10 --smiles_column Canonical 
# python -m utils.assign.assign_formula_parallel -i data/raw/MoNA/positive/output_assign_mona_positive.hdf5 -o data/raw/MoNA/positive/output_assign2_mona_positive.hdf5 --mode FragFormula --chunk_size 100 --num_workers 50 --mass_tolerance 0.01 --timeout_seconds 10 --smiles_column Canonical 
def main():
    parser = argparse.ArgumentParser(description="Run assign_formula in parallel on split MSDataset")
    parser.add_argument("-i", "--input_file", required=True, help="Input MSDataset HDF5 file")
    parser.add_argument("-o", "--output_file", required=True, help="Output file for merged chunks")
    parser.add_argument("--mode", choices=["PossibleFormula", "FragFormula"], required=True, help="Annotation mode")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Number of spectra per chunk")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--mass_tolerance", type=float, default=0.01)
    parser.add_argument("--timeout_seconds", type=int, default=10)
    parser.add_argument("--smiles_column", type=str, default="SMILES")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Step 1: split dataset into chunks
    dataset = MSDataset.from_hdf5(args.input_file)
    temp_dir, chunk_files, output_files = split_dataset(
        dataset, 
        chunk_size=args.chunk_size, 
        column=args.smiles_column,
        return_outputs=True)

    # Step 2: prepare task arguments
    task_args = []
    for in_file, out_file in zip(chunk_files, output_files):
        task_args.append([
            "-i", in_file,
            "-o", out_file,
            "--mode", args.mode,
            "--mass_tolerance", str(args.mass_tolerance),
            "--timeout_seconds", str(args.timeout_seconds),
            "--smiles_column", args.smiles_column,
        ] + (["--resume"] if args.resume else []))

    # Step 3: run in parallel
    print(f"Running {len(task_args)} chunks in parallel with {args.num_workers} workers...")
    run_parallel_subprocesses(SCRIPT_PATH, task_args, max_workers=args.num_workers)

    print("All chunks processed.")
    print("Outputs written to:", args.output_file)

    # Step 4: merge output chunks
    merged_dataset = merge_datasets(output_files, device=dataset.device)
    merged_dataset.to_hdf5(args.output_file)

if __name__ == "__main__":
    main()
