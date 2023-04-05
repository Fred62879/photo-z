
import random

from PIL import Image
import skimage.transform
from torchvision import transforms
from PIL import ImageFilter, ImageOps
#from utils.sdss_dr12_galactic_reddening import SDSSDR12Reddening


# class RandomRotate:
#   def __call__(self, image):
#     return (skimage.transform.rotate(image, np.float32(360*np.random.rand(1)))).astype(np.float32)

# class JitterCrop:
#     def __init__(self, outdim, jitter_lim=None):
#         self.outdim = outdim
#         self.jitter_lim = jitter_lim
#         self.offset = self.outdim//2

#     def __call__(self, image):
#         # print("JC", image.shape, image.dtype)
#         center_x = image.shape[0]//2
#         center_y = image.shape[0]//2
#         if self.jitter_lim:
#             center_x += int(np.random.randint(-self.jitter_lim, self.jitter_lim+1, 1))
#             center_y += int(np.random.randint(-self.jitter_lim, self.jitter_lim+1, 1))

#         return image[(center_x-self.offset):(center_x+self.offset), (center_y-self.offset):(center_y+self.offset)]

# def get_data_loader(params, files_pattern, distributed, is_train, load_specz):
#     if is_train:
#         transform = transforms.Compose([SDSSDR12Reddening(deredden=True),
#                                         RandomRotate(),
#                                         JitterCrop(outdim=params.crop_size, jitter_lim=params.jc_jit_limit),
#                                         transforms.ToTensor()])
#     else:
#         transform = transforms.Compose([SDSSDR12Reddening(deredden=True),
#                                         JitterCrop(outdim=params.crop_size),
#                                         transforms.ToTensor()])

#         dataset = SDSSDataset(params.num_classes, files_pattern, transform, load_specz, True, params.specz_upper_lim)
#         sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
#         dataloader = DataLoader(dataset,
#                                 batch_size=int(params.batch_size) if is_train else int(params.valid_batch_size_per_gpu),
#                                 num_workers=params.num_data_workers,
#                                 shuffle=(sampler is None),
#                                 sampler=sampler,
#                                 drop_last=True,
#                                 pin_memory=torch.cuda.is_available())

#         return dataloader, sampler

class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale), #, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(0.1),
            Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops

class GaussianBlur(object):
    """ Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

class Solarization(object):
    """ Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
