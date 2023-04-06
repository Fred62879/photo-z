
from torchvision import datasets
from dataset.transforms import ImageNetDINOTransform


def ImageNetDataset(**kwargs):
    transform = ImageNetDINOTransform(
        kwargs["global_crops_scale"],
        kwargs["local_crops_scale"],
        kwargs["local_crops_number"],
    )
    print(kwargs["data_path"])
    return datasets.ImageFolder(kwargs["data_path"], transform=transform)
