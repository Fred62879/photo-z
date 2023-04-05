
import random
import numpy as np
#from itertools import zip_longest
from torch.utils.data import Sampler


class PatchWiseSampler(Sampler):
    """ Divide image patches into groups and sample crops randomly within each group.
        @Params:
          data_source: source dataset to sample.
          batch_size (int): max number of crops per training batch.
          num_patches_per_group (int): number of image patches in a group.
        @Return
          yields iter, each value being a batched list of crop index (and/or
            patch switch signal)

        Note: each value of the iter is a actually batch of ids and we always
              guarantee that the patch switch signal, if in the current batch
              is always the first entry.

              Since we rely fully on this sampler for data batching, we need to set
              the batch size of the dataloader to be 1.
              As a result __len__ returns the number of batches
    """
    def __init__(self, data_source, batch_size, num_patches_per_group):
        self.batch_size = batch_size
        self.num_crops = data_source.get_num_crops()
        self.num_patches_per_group = num_patches_per_group
        self.total_num_crops = data_source.get_total_num_crops()

        self.num_patches = len(self.num_crops)
        self.patch_ids = np.arange(self.num_patches)
        self.crop_ids = [
            np.arange(cur_num_crops) for cur_num_crops in self.num_crops
        ]
        self.num_groups = int(np.ceil(self.num_patches / self.num_patches_per_group))

    def __iter__(self):
        np.random.shuffle(self.patch_ids)

        self.indices = []
        for i in range(self.num_groups):
            cur_indices = []
            cur_num_patches = min(
                self.num_patches_per_group,
                self.num_patches - i * self.num_patches_per_group)

            lo = i * self.num_patches_per_group
            cur_patch_ids = self.patch_ids[lo : lo + cur_num_patches]

            # signals dataset to change patch
            change_patch_signal = [-1] + [len(cur_patch_ids)] + list(cur_patch_ids)

            acc_num_crops = 0
            for cur_patch_id in cur_patch_ids:
                patch_crop_ids = self.crop_ids[cur_patch_id] + acc_num_crops
                cur_indices.extend(patch_crop_ids)
                acc_num_crops += self.num_crops[cur_patch_id]

            # mix crops within and across patches in the same group
            random.shuffle(cur_indices)

            # break into batches
            cur_num_batches = int(np.ceil(len(cur_indices) / self.batch_size))
            cur_indices = [
                cur_indices[
                    i * self.batch_size :
                    min( (i+1) * self.batch_size, len(cur_indices) )
                ] for i in range(cur_num_batches)
            ]

            # add signal as the first entry
            cur_indices[0] = change_patch_signal + cur_indices[0]

            self.indices.extend(cur_indices)

        return iter(self.indices)

    def __len__(self) -> int:
        return self.total_num_crops
