from pathlib import Path
import shutil
import typer

app = typer.Typer()

@app.command()
def rename_and_move_for_nnunet(input_dir, output_dir, subdir_name=""):
    """
    Args:
        input_dir: absolute path to the directory with all the nifti cases
        output_dir: absolute path to the directory for nnunet inference
        subdir_name: optional name of subdirectory within case dir
    """
    output_dir = Path(output_dir)
    input_dir = Path(input_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    image_paths = input_dir.rglob("*.nii.gz")
    for image_path in image_paths:
        id_ = image_path.relative_to(input_dir).parts[0]
        if subdir_name in str(image_path):
            output_path = output_dir / (f"{id_}_0000.nii.gz")
            shutil.copy(str(image_path), str(output_path))
