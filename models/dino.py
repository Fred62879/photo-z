
import copy
import torch.nn as nn
import models.vision_transformer as vits

from models.heads import DINOHead
from models.model_utils import has_batchnorms, MultiCropWrapper


class DINO(nn.Module):
    def __init__(self, **kwargs):
        super(DINO, self).__init__()

        self.kwargs = kwargs

        self.init_model()
        self.clip_grad = kwargs["clip_grad"]
        self.freeze_last_layer = kwargs["freeze_last_layer"]

    def init_model(self):
        # print(vits.__dict__.keys())
        self.student = vits.__dict__[self.kwargs["arch"]](
            patch_size=self.kwargs["patch_size"],
            drop_path_rate=self.kwargs["drop_path_rate"],  # stochastic depth
        )
        embed_dim = self.student.embed_dim
        self.student = MultiCropWrapper(self.student, DINOHead(
            embed_dim,
            self.kwargs["out_dim"],
            use_bn=self.kwargs["use_bn_in_head"],
            norm_last_layer=self.kwargs["norm_last_layer"])
        )

        self.teacher = MultiCropWrapper(
            vits.__dict__[self.kwargs["arch"]](patch_size=self.kwargs["patch_size"]),
            DINOHead(embed_dim, self.kwargs["out_dim"], self.kwargs["use_bn_in_head"]),
        )

        if has_batchnorms(self.student):
            self.student = nn.SyncBatchNorm.convert_sync_batchnorm(self.student)
            self.teacher = nn.SyncBatchNorm.convert_sync_batchnorm(self.teacher)

        self.teacher.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False

        arch = self.kwargs["arch"]
        print(f"Student and Teacher are built: they are both {arch} network.")

    def forward(self, images):
        teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
        student_output = student(images)
        return teacher_output, student_output

    def update_student(self, epoch):
        if self.clip_grad:
            param_norms = clip_gradients(self.student, self.clip_grad)
        cancel_gradients_last_layer(epoch, self.student, self.freeze_last_layer)

    def update_teacher(self, total_iterations, momentum_schedule):
        with torch.no_grad():
            m = momentum_schedule[total_iterations] # momentum parameter
            for param_q, param_k in zip(
                    self.student.module.parameters(), self.teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
