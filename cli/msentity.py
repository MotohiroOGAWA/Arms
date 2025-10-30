import click
import os
from arms.cores.MassEntity.msentity.utils.annotate import set_spec_id, MSDataset
from arms.cores.MassEntity.msentity.io import read_msp, read_mgf, write_msp, write_mgf

@click.group()
def main():
    """MSDataset command line tools."""
    pass

def infer_file_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext in ("h5", "hdf5"):
        return "hdf5"
    elif ext in ("msp",):
        return "msp"
    elif ext in ("mgf",):
        return "mgf"
    else:
        raise click.BadParameter(f"Cannot infer file type from extension: {path}")


@main.command("show-record")
@click.argument(
    "input_file",
    type=click.Path(
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True
    ),
)
@click.option(
    "--input-type",
    default="",
    type=click.Choice(["", "hdf5", "msp", "mgf"], case_sensitive=False),
    help="Type of the input dataset. If omitted, inferred from file extension.",
)
def show_record(input_file, input_type):
    """
    Open the dataset and interactively view records by index.
    Example:
        python -m set-specid show-record dataset.h5
    """

    # --- Infer dataset type ---
    if input_type == "":
        input_type = infer_file_type(input_file)

    if not os.path.exists(input_file):
        raise click.FileError(input_file, hint="Input file does not exist.")
    
    click.echo(f"Loading dataset: {input_file} ({input_type})")

    # --- Load dataset once ---
    if input_type == "hdf5":
        dataset = MSDataset.from_hdf5(input_file)
    elif input_type == "msp":
        dataset = read_msp(input_file)
    elif input_type == "mgf":
        dataset = read_mgf(input_file)
    else:
        raise click.BadParameter(f"Unsupported input type: {input_type}")

    click.echo(f"Dataset loaded successfully. Total records: {len(dataset)}")
    click.echo("Type 'exit' or 'quit(q)' to stop.")
    click.echo("-" * 60)

    # --- Interactive loop ---
    while True:
        try:
            user_input = input("Enter record index: ").strip().lower()
            if user_input in ("exit", "quit", "q"):
                click.echo("Exiting interactive mode.")
                break
            if not user_input.isdigit():
                click.echo("Please enter a valid integer index.")
                continue

            index = int(user_input)
            if index < 0 or index >= len(dataset):
                click.echo(f"Index {index} is out of range (0–{len(dataset)-1}).")
                continue

            record = dataset[index]
            click.echo("=" * 60)
            click.echo(f"Record #{index}")
            click.echo("=" * 60)

            click.echo(str(record))

            click.echo("-" * 60)

        except KeyboardInterrupt:
            click.echo("\nInterrupted by user. Exiting.")
            break
        except Exception as e:
            click.echo(f"Error: {e}")




@main.command("set-specid")
@click.argument(
    "input",
    type=click.Path(
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True
    ),
)
@click.option(
    "--input-type",
    default="",
    type=click.Choice(["", "hdf5", "msp", "mgf"], case_sensitive=False),
    help="Type of the input dataset. If omitted, inferred from file extension.",
)
@click.option(
    "-o", "--output",
    default=None,
    type=click.Path(
        exists=False,
        dir_okay=False,
        writable=True,
        resolve_path=True
    ),
    help="Path to save the processed dataset.",
)
@click.option(
    "--output-type",
    default="",
    type=click.Choice(["", "hdf5", "msp", "mgf"], case_sensitive=False),
    help="Type of the output dataset. If omitted, inferred from output file extension.",
)
@click.option(
    "--prefix",
    default="",
    help="Prefix for generated SpecIDs.",
)
@click.option(
    "-f", "--force",
    is_flag=False,
    help="Force overwrite of output file if it exists.",
)
def set_specid(input, input_type, output, output_type, prefix, force):
    """
    Process an MSDataset: load INPUT (HDF5/MSP/MGF),
    assign SpecID, and save to OUTPUT.
    """

    # --- Infer file types if not provided ---
    if input_type == "":
        input_type = infer_file_type(input)
    if output_type == "":
        output_type = infer_file_type(output if output else input)

    if not os.path.exists(input):
        raise click.FileError(input, hint="Input file does not exist.")

    # --- Handle case when output path is not provided ---
    if output is None:
        click.echo(f"No output specified. The input file will be overwritten: {input}")
        if not force:
            # Ask user for confirmation before overwriting input file
            if not click.confirm("Do you want to continue and overwrite the input file?"):
                click.echo("Operation cancelled.")
                return
        output = input  # User confirmed overwrite or force flag is set
    else:
        click.echo(f"Input:  {input} ({input_type})")
        click.echo(f"Output: {output} ({output_type})")
        click.echo(f"Prefix: {prefix}")

    # --- Load dataset ---
    if input_type == "hdf5":
        dataset = MSDataset.from_hdf5(input)
    elif input_type == "msp":
        dataset = read_msp(input)
    elif input_type == "mgf":
        dataset = read_mgf(input)
    else:
        raise click.BadParameter(f"Unsupported input type: {input_type}")
    
    # --- Set SpecID ---
    success = set_spec_id(dataset, prefix)
    if not success:
        click.echo("SpecID assignment skipped.")
    else:
        click.echo("SpecID assigned successfully.")
    
    # --- Save dataset ---
    if output_type == "hdf5":
        dataset.to_hdf5(output)
    elif output_type == "msp":
        write_msp(dataset, output)
    elif output_type == "mgf":
        write_mgf(dataset, output)

if __name__ == "__main__":
    main()
