
import torch
import torch.nn as nn
import models.vision_transformer as vits

from models.heads import DINOHead
from models.model_utils import load_model_prefixed_state
from models.model_utils import clip_gradients, has_batchnorms, MultiCropWrapper


class DINOz(nn.Module):
    def __init__(self, pre_trained_model_fname, **kwargs):
        super(DINOz, self).__init__()

        self.kwargs = kwargs
        self.init_model(pre_trained_model_fname)

    def init_model(self, fname):
        vit = vits.__dict__[self.kwargs["arch"]](
            patch_size=self.kwargs["patch_size"],
            in_chans=self.kwargs["in_chans"],
            # drop_path_rate=self.kwargs["drop_path_rate"],  # stochastic depth
        )

        state = torch.load(fname)["model_state_dict"]
        # for n,p in vit.named_parameters(): print(n)
        # for n in state.keys(): print(n)

        load_model_prefixed_state(state, vit, "student.backbone.")
        for p in vit.parameters():
            p.requires_grad = False

        embed_dim = vit.embed_dim
        self.model = MultiCropWrapper(vit, DINOHead(
            embed_dim,
            self.kwargs["out_dim"],
            use_bn=self.kwargs["use_bn_in_head"],
            # norm_last_layer=self.kwargs["norm_last_layer"])
        ))

        # if has_batchnorms(self.student):
        #     self.teacher = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        arch = self.kwargs["arch"]
        print(f"Redshift prediction model is built as {arch} network.")

    def forward(self, images):
        # images BCHW
        logits = self.model(images) #
        return logits
