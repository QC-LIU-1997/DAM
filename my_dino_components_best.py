from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

import math
import warnings

from torch.nn.functional import cosine_similarity

class rawdino_loss(nn.Module):
    def __init__(self):
        super(rawdino_loss, self).__init__()
    def forward(self, anchor, positive, negetive):
        t_out = F.softmax(anchor, dim=-1)
        s_nonde_out = F.log_softmax(positive, dim=-1)
        s_de_out = F.log_softmax(negetive, dim=-1)
        loss = -torch.einsum("tbd,sbd->ts", t_out, s_nonde_out)
        # # print('raw',loss)
        # print(loss.shape)
        # loss.fill_diagonal_(0)
        # print('new',loss)

        # number of loss terms, ignoring the diagonal
        # n_terms = loss.numel() - loss.diagonal().numel()
        batch_size = t_out.shape[1]
        return loss.sum() / (batch_size)

# init pow 2 !!!

soft_act = nn.ELU(alpha=1.0, inplace=False)

class triplet_mae_loss(nn.Module):
    def __init__(self,margin):
        super(triplet_mae_loss, self).__init__()
        self.margin = margin
    def forward(self, anchor, positive, negative):
        # print(anchor.shape,positive.shape,negative.shape)
        pos_dist = torch.abs(anchor - positive).sum(-1)
        neg_dist = torch.abs(anchor - negative).sum(-1)
        # print(pos_dist.shape,neg_dist.shape)
        loss = F.relu(pos_dist - neg_dist + self.margin).nan_to_num(nan = self.margin)
        return loss.mean()

class triplet_loss(nn.Module):
    def __init__(self,margin):
        super(triplet_loss, self).__init__()
        self.margin = margin
    def forward(self, anchor, positive, negative):
        # print(anchor.shape,positive.shape,negative.shape)
        pos_dist = (anchor - positive).pow(2).sum(-1)
        neg_dist = (anchor - negative).pow(2).sum(-1)
        # print(pos_dist.shape,neg_dist.shape)
        loss = F.relu(pos_dist - neg_dist + self.margin).nan_to_num(nan = self.margin)
        return loss.mean()
    
class tritriplet_loss(nn.Module):
    def __init__(self,margin):
        super(tritriplet_loss, self).__init__()
        self.margin = margin
    def forward(self, anchor, positive, negative):
        # print(anchor.shape,positive.shape,negative.shape)
        pos_dist,_ = torch.max((anchor - positive).pow(2).sum(-1), dim = 0)
        neg_dist,_ = torch.min((anchor - negative).pow(2).sum(-1), dim = 0)
        # print(pos_dist.unsqueeze(1) - neg_dist.unsqueeze(1) + self.margin)
        # print(F.relu(pos_dist.unsqueeze(1) - neg_dist.unsqueeze(1) + self.margin))
        loss = F.relu(pos_dist.unsqueeze(1) - neg_dist.unsqueeze(1) + self.margin).nan_to_num(nan = self.margin)
        return loss.mean()


class dual_loss(nn.Module):
    def __init__(self,margin = None):
        super(dual_loss, self).__init__()
        self.margin = margin
        self.dual = nn.MSELoss()
    def forward(self, anchor, positive, negative):
        loss =  self.dual(F.normalize(anchor, dim=-1),F.normalize(positive, dim=-1))

        return loss.mean()

class triplet_cos_loss(nn.Module):
    def __init__(self,margin):
        super(triplet_cos_loss, self).__init__()
        self.margin = margin
    def forward(self, anchor, positive, negative):
        # print(anchor, positive,negative)
        pos_sim = cosine_similarity(anchor, positive)
        neg_sim = cosine_similarity(anchor, negative)
        # loss = F.relu(neg_sim  - pos_sim + self.margin)
        # print(pos_sim,neg_sim)
        # temp_flag = (neg_sim  - pos_sim + self.margin).nan_to_num(nan = self.margin)
        loss = F.relu(neg_sim  - pos_sim + self.margin).nan_to_num(nan = self.margin)
        # print(loss)
        return loss.mean()

class My_DINOLoss(nn.Module):
    """
    Implementation of the loss described in 'Emerging Properties in
    Self-Supervised Vision Transformers'. [0]

    This implementation follows the code published by the authors. [1]
    It supports global and local image crops. A linear warmup schedule for the
    teacher temperature is implemented to stabilize training at the beginning.
    Centering is applied to the teacher output to avoid model collapse.

    - [0]: DINO, 2021, https://arxiv.org/abs/2104.14294
    - [1]: https://github.com/facebookresearch/dino

    Attributes:
        output_dim:
            Dimension of the model output.
        warmup_teacher_temp:
            Initial value of the teacher temperature. Should be decreased if the
            training loss does not decrease.
        teacher_temp:
            Final value of the teacher temperature after linear warmup. Values
            above 0.07 result in unstable behavior in most cases. Can be
            slightly increased to improve performance during finetuning.
        warmup_teacher_temp_epochs:
            Number of epochs for the teacher temperature warmup.
        student_temp:
            Temperature of the student.
        center_momentum:
            Momentum term for the center calculation.

    Examples:

        >>> # initialize loss function
        >>> loss_fn = DINOLoss(128)
        >>>
        >>> # generate a view of the images with a random transform
        >>> view = transform(images)
        >>>
        >>> # embed the view with a student and teacher model
        >>> teacher_out = teacher(view)
        >>> student_out = student(view)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn([teacher_out], [student_out], epoch=0)

    """

    def __init__(
        self,
        device,
        output_dim: int = 65536,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.04,
        warmup_teacher_temp_epochs: int = 30,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
        tri_type = 'TMWDL',
        margin = None

    ):
        super().__init__()
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum

        self.register_buffer("center", torch.zeros(1, 1, output_dim))
        self.center = self.center.to(device)
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = torch.linspace(
            start=warmup_teacher_temp,
            end=teacher_temp,
            steps=warmup_teacher_temp_epochs,
        )
        if tri_type == 'tri':
            self.triplet = triplet_loss(margin=margin)
        elif tri_type == 'tritri':
            self.triplet = tritriplet_loss(margin=margin)
        elif tri_type == 'tricos':
            self.triplet = triplet_cos_loss(margin=margin)
        elif tri_type == 'rawdino':
            self.triplet = rawdino_loss()
        elif tri_type == 'dual':
            self.triplet = dual_loss()
        elif tri_type == 'trimae':
            self.triplet = triplet_mae_loss(margin=margin)

    def forward(
        self,
        teacher_out: List[torch.Tensor],
        student_nonde_out: List[torch.Tensor],
        student_de_out: List[torch.Tensor],
        epoch: int,
    ) -> torch.Tensor:
        """Cross-entropy between softmax outputs of the teacher and student
        networks.

        Args:
            teacher_out:
                List of view feature tensors from the teacher model. Each
                tensor is assumed to contain features from one view of the batch
                and have length batch_size.
            student_out:
                List of view feature tensors from the student model. Each tensor
                is assumed to contain features from one view of the batch and
                have length batch_size.
            epoch:
                The current training epoch.

        Returns:
            The average cross-entropy loss.

        """
        # get teacher temperature
        if epoch < self.warmup_teacher_temp_epochs:
            teacher_temp = self.teacher_temp_schedule[epoch]
        else:
            teacher_temp = self.teacher_temp
        

        teacher_out = torch.stack(teacher_out)
        student_nonde_out = torch.stack(student_nonde_out)
        student_de_out = torch.stack(student_de_out)

        anchor = teacher_out / teacher_temp
        positive = student_nonde_out / self.student_temp
        negetive = student_de_out / self.student_temp

        loss = self.triplet(anchor, positive, negetive)

        # print('teacher_out',teacher_out.shape)
        # print('t_out',t_out.shape)
        # print('student_out',student_nonde_out.shape)
        # print('s_out',s_nonde_out.shape)
        # calculate feature similarities where:
        # b = batch_size, t = n_views_teacher, s = n_views_student, d = output_dim
        # the diagonal is ignored as it contains features from the same views
        self.update_center(teacher_out)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_out: torch.Tensor) -> None:
        """Moving average update of the center used for the teacher output.

        Args:
            teacher_out:
                Stacked output from the teacher model.

        """
        batch_center = torch.mean(teacher_out, dim=(0, 1), keepdim=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center = batch_center / dist.get_world_size()

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )



def _no_grad_trunc_normal(
    tensor: torch.Tensor,
    mean: float,
    std: float,
    a: float,
    b: float,
) -> torch.Tensor:
    """Initializes the input tensor with a truncated normal distribution.

    This method is based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf

    Args:
        tensor:
            The tensor to initialize.
        mean:
            Mean of the distribution.
        std:
            Standard deviation of the distribution.
        a:
            Minimum value of the distribution, values below will be clamped.
        b:
            Maximum value of the distribution, values above will be clamped.

    """

    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

class ProjectionHead(nn.Module):
    """Base class for all projection and prediction heads.

    Args:
        blocks:
            List of tuples, each denoting one block of the projection head MLP.
            Each tuple reads (in_features, out_features, batch_norm_layer,
            non_linearity_layer).

    Examples:
        >>> # the following projection head has two blocks
        >>> # the first block uses batch norm an a ReLU non-linearity
        >>> # the second block is a simple linear layer
        >>> projection_head = ProjectionHead([
        >>>     (256, 256, nn.BatchNorm1d(256), nn.ReLU()),
        >>>     (256, 128, None, None)
        >>> ])

    """

    def __init__(
        self, blocks: List[Tuple[int, int, Optional[nn.Module], Optional[nn.Module]]]
    ) -> None:
        super(ProjectionHead, self).__init__()

        layers: List[nn.Module] = []
        for input_dim, output_dim, batch_norm, non_linearity in blocks:
            use_bias = not bool(batch_norm)
            layers.append(nn.Linear(input_dim, output_dim, bias=use_bias))
            if batch_norm:
                layers.append(batch_norm)
            if non_linearity:
                layers.append(non_linearity)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Computes one forward pass through the projection head.

        Args:
            x:
                Input of shape bsz x num_ftrs.

        """
        projection: Tensor = self.layers(x)
        return projection

class DINOProjectionHead(ProjectionHead):
    """Projection head used in DINO.

    "The projection head consists of a 3-layer multi-layer perceptron (MLP)
    with hidden dimension 2048 followed by l2 normalization and a weight
    normalized fully connected layer with K dimensions, which is similar to the
    design from SwAV [1]." [0]

    - [0]: DINO, 2021, https://arxiv.org/abs/2104.14294
    - [1]: SwAV, 2020, https://arxiv.org/abs/2006.09882

    Attributes:
        input_dim:
            The input dimension of the head.
        hidden_dim:
            The hidden dimension.
        bottleneck_dim:
            Dimension of the bottleneck in the last layer of the head.
        output_dim:
            The output dimension of the head.
        batch_norm:
            Whether to use batch norm or not. Should be set to False when using
            a vision transformer backbone.
        freeze_last_layer:
            Number of epochs during which we keep the output layer fixed.
            Typically doing so during the first epoch helps training. Try
            increasing this value if the loss does not decrease.
        norm_last_layer:
            Whether or not to weight normalize the last layer of the DINO head.
            Not normalizing leads to better performance but can make the
            training unstable.

    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        output_dim: int = 65536,
        batch_norm: bool = False,
        freeze_last_layer: int = -1,
        norm_last_layer: bool = True,
    ):
        bn = nn.BatchNorm1d(hidden_dim) if batch_norm else None

        super().__init__(
            [
                (input_dim, hidden_dim, bn, nn.GELU()),
                (hidden_dim, hidden_dim, bn, nn.GELU()),
                (hidden_dim, bottleneck_dim, None, None),
            ]
        )
        self.apply(self._init_weights)
        self.freeze_last_layer = freeze_last_layer
        self.last_layer = nn.Linear(bottleneck_dim, output_dim, bias=False)
        self.last_layer = nn.utils.weight_norm(self.last_layer)
        # Tell mypy this is ok because fill_ is overloaded.
        self.last_layer.weight_g.data.fill_(1)  # type: ignore

        # Option to normalize last layer.
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def cancel_last_layer_gradients(self, current_epoch: int) -> None:
        """Cancel last layer gradients to stabilize the training."""
        if current_epoch >= self.freeze_last_layer:
            return
        for param in self.last_layer.parameters():
            param.grad = None

    def _init_weights(self, module: nn.Module) -> None:
        """Initializes layers with a truncated normal distribution."""
        if isinstance(module, nn.Linear):
            _no_grad_trunc_normal(
                module.weight,
                mean=0,
                std=0.2,
                a=-2,
                b=2,
            )
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """Computes one forward pass through the head."""
        x = self.layers(x)
        # l2 normalization
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x