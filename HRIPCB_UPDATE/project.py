import os
import shutil
import random
from pathlib import Path

def split_dataset(image_root, annotation_root, output_root, split_ratios=(0.7, 0.2, 0.1)):
    """
    Split the dataset into train, test, and validation sets.
    Args:
        image_root: Path to the folder containing class subfolders of images.
        annotation_root: Path to the folder containing class subfolders of annotations.
        output_root: Path to the output folder where train, test, val folders will be created.
        split_ratios: A tuple (train_ratio, test_ratio, val_ratio) summing up to 1.0.
    """
    assert sum(split_ratios) == 1.0, "Split ratios must sum up to 1.0."

    # Create output directories
    for split in ['train', 'test', 'val']:
        os.makedirs(Path(output_root) / split / 'images', exist_ok=True)
        os.makedirs(Path(output_root) / split / 'annotations', exist_ok=True)

    classes = os.listdir(image_root)
    file_pairs = []

    # Collect all image and annotation file pairs
    for cls in classes:
        image_dir = Path(image_root) / cls
        annotation_dir = Path(annotation_root) / cls

        # Get all image and annotation files
        image_files = sorted(image_dir.glob("*"))
        annotation_files = sorted(annotation_dir.glob("*"))

        # Ensure image and annotation files match
        assert len(image_files) == len(annotation_files), f"Mismatch in number of files for class {cls}."
        
        # Add pairs of image and annotation files
        file_pairs.extend(zip(image_files, annotation_files))

    # Shuffle the data
    random.shuffle(file_pairs)

    # Calculate split sizes
    total_files = len(file_pairs)
    train_split = int(total_files * split_ratios[0])
    test_split = train_split + int(total_files * split_ratios[1])

    # Split the data into train, test, and validation
    train_files = file_pairs[:train_split]
    test_files = file_pairs[train_split:test_split]
    val_files = file_pairs[test_split:]

    # Function to copy files to the output folder
    def copy_files(file_pairs, split):
        for image_file, annotation_file in file_pairs:
            shutil.copy(image_file, Path(output_root) / split / 'images' / image_file.name)
            shutil.copy(annotation_file, Path(output_root) / split / 'annotations' / annotation_file.name)

    # Copy files to the respective splits
    copy_files(train_files, 'train')
    copy_files(test_files, 'test')
    copy_files(val_files, 'val')

# Paths to your dataset
image_root = "C:/Users/eslam/Downloads/PCB_DATASET/PCB_DATASET/images"
annotation_root = "C:/Users/eslam/Downloads/PCB_DATASET/PCB_DATASET/Annotations"
output_root = "D:/Output"

# Split the dataset
split_dataset(image_root, annotation_root, output_root, split_ratios=(0.7, 0.2, 0.1))