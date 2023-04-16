
import random
import numpy as np
import logging as log
import skimage.transform as sk_transform

#from PIL import Image
from torchvision import transforms
from scipy.ndimage import gaussian_filter
#from PIL import ImageFilter, ImageOps


class SpecZBin:
    def __init__(self, specz_upper_lim, num_specz_bins):
        self.specz_upper_lim = specz_upper_lim
        self.num_specz_bins = num_specz_bins

    def __call__(self, specz):
        specz_bin = torch.ByteTensor(
            specz // (self.specz_upper_lim / self.num_specz_bins)
        )
        return specz_bin

class RandomRotate:
    def __init__(self, mode="wrap"):
        self.mode = mode

    def __call__(self, image):
        bsz = image.shape[0]
        degs = np.float32(360*np.random.rand(bsz))
        for i in range(bsz):
            image[i] = sk_transform.rotate(image[i], degs[i], mode=self.mode)
        return image
        # return (sk_transform.rotate(image, deg)).astype(np.float32)

class JitterCrop:
    def __init__(self, outdim, jitter_lim=None):
        self.outdim = outdim
        self.jitter_lim = jitter_lim
        self.offset = self.outdim//2

    def __call__(self, image):
        """ Batched cropping operation
            @Param
               image [bsz,nbands,sz,sz]
            @Return
               image [bsz,nbands,offset*2,offset*2]
        """
        # image = image[:,:3,:4,:4]
        # self.offset = 1
        # self.jitter_lim = 1
        # print("JC", image.shape, image.dtype)

        bsz, nbands, sz, _ = image.shape
        center_x = image.shape[-2]//2
        center_y = image.shape[-1]//2
        if self.jitter_lim:
            center_x += np.random.randint(-self.jitter_lim, self.jitter_lim+1, bsz)
            center_y += np.random.randint(-self.jitter_lim, self.jitter_lim+1, bsz)
        else:
            center_x = [center_x]
            center_y = [center_y]

        center_x = np.array(center_x)
        center_y = np.array(center_y)

        x_indices = np.vstack(np.arange(start, end) for start, end in zip(
            center_x-self.offset, center_x+self.offset
        ))
        x_indices = np.tile(x_indices[:,None,:,None], (1,nbands,1,1))
        image = np.take_along_axis(image, x_indices, -2)

        y_indices = np.vstack(np.arange(start, end) for start, end in zip(
            center_y-self.offset, center_y+self.offset
        ))
        y_indices = np.tile(y_indices[:,None,None], (1,nbands,1,1))
        image = np.take_along_axis(image, y_indices, -1)
        return image

class GaussianBlur(object):
    """ Apply Gaussian Noise to the image.
    """
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, imgs):
        n, nbands, _, _ = imgs.shape

        for i in range(n):
            for j in range(nbands):
                imgs[i][j] = gaussian_filter(imgs[i][j], sigma=self.sigma[j])
        return imgs

class SDSSDR12Reddening:
    def __init__(self, deredden = False, redden_aug = False, ebv_max = 0.5):
        self.R = np.array([4.239, 3.303, 2.285, 1.698, 1.263])
        self.R_dr12 = np.array([5.155, 3.793, 2.751, 2.086, 1.479])
        self.deredden = deredden
        self.redden_aug = redden_aug # apply reddening augmentation
        self.ebv_max = ebv_max

    def __call__(self, data):
        if type(data)==list:
            image = data[0]
            if self.deredden:
                dr12_ext = data[1]
                sfd_ebv = np.mean(dr12_ext/self.R_dr12)
                true_ext = self.R * sfd_ebv
                image = np.float32(
                    image * (10.**(true_ext/2.5))) # deredden image
        else:
            image = data
            if self.deredden:
                log.error("Dereddening requested but no ebv value passed from dataset loader")
                exit(1)

        if self.redden_aug:
            new_ebv = np.random.uniform(0, self.ebv_max)
            image = np.float32(image*(10.**(-self.R*new_ebv/2.5)))

        return image

class RedshiftDINOTransform(object):
    def __init__(self, **kwargs):
        # first global crop
        self.global_transform1 = transforms.Compose([
            JitterCrop(kwargs["dino_global_crop_dim"]),
            #GaussianBlur(1.0),
            #transforms.ToTensor()
        ])
        # second global crop
        self.global_transform2 = transforms.Compose([
            #SDSSDR12Reddening(deredden=True),
            JitterCrop(kwargs["dino_global_crop_dim"]),
            #GaussianBlur(1.0)
            #transforms.ToTensor(),
        ])
        # transformation for the local small crops
        self.local_crops_number = kwargs["dino_num_local_crops"]
        self.local_transform = transforms.Compose([
            JitterCrop(kwargs["dino_local_crop_dim"], kwargs["dino_jitter_lim"]),
            RandomRotate(kwargs["dino_rotate_mode"]),
            #GaussianBlur(p=0.5),
            #transforms.ToTensor(),
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transform1(image))
        crops.append(self.global_transform2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(image))
        return crops

# class ImageNetDINOTransform(object):
#     def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
#         flip_and_color_jitter = transforms.Compose([
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.RandomApply(
#                 [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
#                 p=0.8
#             ),
#             transforms.RandomGrayscale(p=0.2),
#         ])
#         normalize = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#         ])

#         # first global crop
#         self.global_transfo1 = transforms.Compose([
#             transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
#             flip_and_color_jitter,
#             GaussianBlur(1.0),
#             normalize,
#         ])
#         # second global crop
#         self.global_transfo2 = transforms.Compose([
#             transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
#             flip_and_color_jitter,
#             GaussianBlur(0.1),
#             Solarization(0.2),
#             normalize,
#         ])
#         # transformation for the local small crops
#         self.local_crops_number = local_crops_number
#         self.local_transfo = transforms.Compose([
#             transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
#             flip_and_color_jitter,
#             GaussianBlur(p=0.5),
#             normalize,
#         ])

#     def __call__(self, image):
#         crops = []
#         crops.append(self.global_transfo1(image))
#         crops.append(self.global_transfo2(image))
#         for _ in range(self.local_crops_number):
#             crops.append(self.local_transfo(image))
#         return crops

# class GaussianBlur(object):
#     """ Apply Gaussian Blur to the PIL image.
#     """
#     def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
#         self.prob = p
#         self.radius_min = radius_min
#         self.radius_max = radius_max

#     def __call__(self, img):
#         do_it = random.random() <= self.prob
#         if not do_it:
#             return img

#         return img.filter(
#             ImageFilter.GaussianBlur(
#                 radius=random.uniform(self.radius_min, self.radius_max)
#             )
#         )

# class Solarization(object):
#     """ Apply Solarization to the PIL image.
#     """
#     def __init__(self, p):
#         self.p = p

#     def __call__(self, img):
#         if random.random() < self.p:
#             return ImageOps.solarize(img)
#         else:
#             return img
