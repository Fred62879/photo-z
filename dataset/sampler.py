

class PatchWiseSampler(BatchSampler):
    """Samples either batch_size images or batches num_objs_per_batch objects.

    Args:
        data_source (list): contains tuples of (img_id).
        batch_size (int): batch size.
        num_objs_per_batch (int): number of objects in a batch.
    Return
        yields the batch_ids/image_ids/image_indices

    """

    def __init__(self, data_source, num_patches_per_batch, num_crops_per_batch, drop_last=False):
        self.data_source = data_source
        self.num_crops = data_source.get_num_crops()
        self.num_patches = len(self.num_crops)
        self.fits_fnames = data_source.get_fits_fnames()

        self.drop_last = drop_last
        self.num_crops_per_batch = num_crops_per_batch
        self.num_patches_per_batch = num_patches_per_batch

        self.cur_patch_ids = None
        self.

    def __iter__(self):

        if self.cur_patch_ids is None:
            self.cur_patch_ids

        batch = []
        img_counts_id = 0
        for idx, (k, s) in enumerate(self.data_source.iteritems()):
            if len(batch) < self.batches[img_counts_id] and idx < len(self.data_source):
                batch.append(s)
            elif len(batch) == self.batches[img_counts_id]:
                gc.collect()
                yield batch
                batch = []
                if img_counts_id < self.batch_count - 1:
                    img_counts_id += 1
                else:
                    break

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size
