"""Data loader."""

import os
import pathlib
from typing import Tuple, Dict, List
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folder names in a target directory.
    
    Assumes target directory is in standard image classification format.

    Args:
        directory (str): target directory to load classnames from.

    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))
    
    Example:
        find_classes("food_images/train")
        >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    # 1. Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    
    # 2. Raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
        
    # 3. Create a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

class CustomDataset(Dataset):
    """Custom Dataset."""

    def __init__(self, targ_dir, transforms=None):
        self.file_list = list(pathlib.Path(targ_dir).glob("*/*.jpg"))
        self.transforms = transforms
        self.classes, self.class_to_idx = find_classes(targ_dir)
    
    def __len__(self):
        return len(self.file_list)

    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path) 

    def __getitem__(self, index: int):
        img = self.load_image(index)
        class_name  = self.paths[index].parent.name # expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]

        if self.transforms:
            return self.transforms(img), class_idx
        return img, class_idx


def get_data_transform(image_size):
    """Get transform."""
    train_transforms =  transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=30),
        # transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    return train_transforms, val_transforms


def get_data_loader(
    train_data: CustomDataset,
    valid_data: CustomDataset,
    batch_size: int,
    shuffle: bool = True
):
    """Get data loader."""
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=args.batch_size, shuffle=True)
    return train_loader, valid_loader
