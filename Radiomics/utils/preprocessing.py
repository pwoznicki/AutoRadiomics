import os
from pathlib import Path
import shutil
import typer
from nipype.interfaces.dcm2nii import Dcm2niix

app = typer.Typer()

def get_dcm2niix_converter(dicom_dir, save_dir):
    converter = Dcm2niix()
    converter.inputs.source_dir = str(dicom_dir)
    converter.inputs.output_dir = str(save_dir)
    converter.inputs.single_file = True
    #converter.inputs.out_filename = "%d"
    converter.inputs.merge_imgs = True

    return converter

@app.command()
def dicom_to_nifti(input_dir: str, output_dir: str, subdir_name: str = ""):
    """
    Args:
        input_dir: absolute path to the directory with all the cases containing
                dicoms
        output_dir: absolute path to the directory where to save nifties
        subdir_name: optional name of subdirectory within case dir
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    for id_ in os.listdir(input_dir):
        dicom_dir = input_dir / id_ / subdir_name
        save_dir = output_dir / id_ / subdir_name
        save_dir.mkdir(exist_ok=True)
        assert dicom_dir.exists()

        converter = get_dcm2niix_converter(dicom_dir, save_dir)
        converter.run()

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
