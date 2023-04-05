
from torchvision import datasets
from dataset.transforms import DataAugmentationDINO


def ImageNetDataset(**kwargs):
    transform = DataAugmentationDINO(
        kwargs["global_crops_scale"],
        kwargs["local_crops_scale"],
        kwargs["local_crops_number"],
    )
    print(kwargs["data_path"])
    return datasets.ImageFolder(kwargs["data_path"], transform=transform)
