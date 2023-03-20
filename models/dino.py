
import copy
import torch.nn as nn

import sys
sys.path.insert(0, './models')
from model_utils import has_batchnorms, MultiCropWrapper


#from lightly.models.modules import DINOProjectionHead
#from lightly.models.utils import deactivate_requires_grad, update_momentum

# class DINO(nn.Module):
#     def __init__(self, backbone, input_dim):
#         super(DINO, self).__init__()
#         self.student_backbone = backbone
#         self.student_head = DINOProjectionHead(
#             input_dim, 512, 64, 2048, freeze_last_layer=1
#         )
#         self.teacher_backbone = copy.deepcopy(backbone)
#         self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
#         deactivate_requires_grad(self.teacher_backbone)
#         deactivate_requires_grad(self.teacher_head)

#     def forward(self, x):
#         y = self.student_backbone(x).flatten(start_dim=1)
#         z = self.student_head(y)
#         return z

#     def forward_teacher(self, x):
#         y = self.teacher_backbone(x).flatten(start_dim=1)
#         z = self.teacher_head(y)
#         return z

class DINO(nn.Module):
    def __init__(self, **kwargs):
        super(DINO, self).__init__()

        self.init_model()
        self.clip_grad = kwargs["clip_student_grad"]
        self.freeze_last_layer = kwargs["freeze_last_layer"]

    def init_model(self):
        self.student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        embed_dim = self.student.embed_dim
        self.student = MultiCropWrapper(self.student, DINOHead(
            embed_dim,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer)
        )

        self.teacher = MultiCropWrapper(
            vits.__dict__[args.arch](patch_size=args.patch_size),
            DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
        )

        if has_batchnorms(self.student):
            self.student = nn.SyncBatchNorm.convert_sync_batchnorm(self.student)
            self.teacher = nn.SyncBatchNorm.convert_sync_batchnorm(self.teacher)

        self.teacher.load_state_dict(self.student.module.state_dict())
        for p in teacher.parameters():
            p.requires_grad = False

        print(f"Student and Teacher are built: they are both {args.arch} network.")

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
