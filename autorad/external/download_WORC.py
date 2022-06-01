# Adapted from https://github.com/MStarmans91/WORCDatabase/blob/
# development/datadownloader.py

import logging
import os
import shutil
from glob import glob

# Original license:
# Copyright 2016-2021 Biomedical Imaging Group Rotterdam, Departments of
# Medical Informatics and Radiology, Erasmus MC, Rotterdam, The Netherlands
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pandas as pd
import xnat

logging.getLogger("xnat").setLevel(logging.WARNING)

valid_datasets = ["Lipo", "Desmoid", "GIST", "Liver", "CRLM", "Melanoma"]


def download_subject(project, subject, datafolder, session, verbose=False):
    """Download data of a single XNAT subject."""
    download_counter = 0
    resource_labels = list()
    for e in subject.experiments:
        resmap = {}
        experiment = subject.experiments[e]

        for s in experiment.scans:
            scan = experiment.scans[s]
            print(
                ("\tDownloading patient {}, experiment {}, scan {}.").format(
                    subject.label, experiment.label, scan.id
                )
            )
            for res in scan.resources:
                resource_label = scan.resources[res].label
                if resource_label == "NIFTI":
                    # Create output directory
                    outdir = os.path.join(datafolder, f"{subject.label}")
                    os.makedirs(outdir, exist_ok=True)

                    resmap[resource_label] = scan
                    # print(f'resource is {resource_label}')
                    scan.resources[res].download_dir(outdir)
                    resource_labels.append(resource_label)
                    download_counter += 1

    # Parse resources and throw warnings if they not meet the requirements
    subject_name = subject.label
    if download_counter == 0:
        print(
            f"\t[WARNING] Skipping subject {subject_name}: no (suitable) resources found."
        )
        return False

    if "NIFTI" not in resource_labels:
        print(
            f"\t[WARNING] Skipping subject {subject_name}: no NIFTI resources found."
        )
        return False

    # Reorder files to a easier to read structure
    NIFTI_files = glob(
        os.path.join(
            outdir,
            "*",
            "scans",
            "*",
            "resources",
            "NIFTI",
            "files",
            "*.nii.gz",
        )
    )
    for NIFTI_file in NIFTI_files:
        basename = os.path.basename(NIFTI_file)
        shutil.move(NIFTI_file, os.path.join(outdir, basename))

    for folder in glob(os.path.join(outdir, "*")):
        if os.path.isdir(folder):
            shutil.rmtree(folder)

    return True


def download_project(
    xnat_url,
    datafolder,
    nsubjects="all",
    verbose=False,
    dataset="all",
):
    """Download data of full XNAT project."""
    # Connect to XNAT and retreive project
    labels_df_path = os.path.join(datafolder, "labels.csv")
    project_name = "worc"
    with xnat.connect(xnat_url) as session:
        project = session.projects[project_name]

        os.makedirs(datafolder, exist_ok=True)

        subjects_len = len(project.subjects)
        if nsubjects != "all":
            nsubjects = min(nsubjects, subjects_len)

        subjects_counter = 1
        downloaded_subjects_counter = 0
        labels = {}
        for subject_name in project.subjects:
            s = project.subjects[subject_name]
            if dataset != "all":
                # Check if patient belongs to required dataset
                subject_dataset = s.fields["dataset"]
                if subject_dataset != dataset:
                    continue

            # print(f"Processing on subject {subjects_counter}/{subjects_len}")
            subjects_counter += 1

            subject_diagnosis = int(s.fields["diagnosis_binary"])
            labels[s.label] = subject_diagnosis

            success = download_subject(
                project_name, s, datafolder, session, verbose
            )
            if success:
                downloaded_subjects_counter += 1

            # Stop downloading if we have reached the required number of subjects
            if downloaded_subjects_counter == nsubjects:
                break
        # Disconnect the session
        session.disconnect()
        if nsubjects != "all":
            if downloaded_subjects_counter < nsubjects:
                raise ValueError(
                    f"Number of subjects downloaded {downloaded_subjects_counter} is smaller than the number required {nsubjects}."
                )

        print("Done downloading!")

        label_df = pd.DataFrame(
            {
                "patient_ID": list(labels.keys()),
                "diagnosis": list(labels.values()),
            }
        )
        label_df.to_csv(labels_df_path, index=False)


def download_WORCDatabase(
    dataset=None,
    data_folder=None,
    n_subjects="all",
):
    """Download a dataset from the WORC Database.
    Download all Nifti images and segmentations from a dataset from the WORC
    database from https://xnat.bmia.nl/data/projects/worc
    dataset: string, default None
        If None, download the full XNAT project. If string, download one
        of the six datasets. Valid values: Lipo, Desmoid, GIST, Liver, CRLM,
        Melanoma
    """
    # Check if dataset is valid
    if dataset not in valid_datasets:
        raise KeyError(
            f"{dataset} is not a valid dataset, should be one of {valid_datasets}."
        )

    if data_folder is None:
        # Download data to path in which this script is located + Data
        cwd = os.getcwd()
        datafolder = os.path.join(cwd, "Data")
        if not os.path.exists(datafolder):
            os.makedirs(datafolder)

    xnat_url = "https://xnat.bmia.nl"
    download_project(
        xnat_url,
        data_folder,
        nsubjects=n_subjects,
        verbose=False,
        dataset=dataset,
    )
