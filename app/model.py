import torch
from torch import nn
from torchvision.models import resnet18


def build_model():
    model = resnet18(weights=None, num_classes=10, norm_layer=lambda n: nn.GroupNorm(32, n))
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def flatten_parameters(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).contiguous()


def load_flat_parameters(model, flat):
    offset = 0
    with torch.no_grad():
        for parameter in model.parameters():
            size = parameter.numel()
            parameter.copy_(flat[offset:offset + size].view_as(parameter))
            offset += size


def flatten_gradients(model):
    return torch.cat(
        [
            torch.zeros_like(p).reshape(-1)
            if p.grad is None
            else p.grad.detach().reshape(-1)
            for p in model.parameters()
        ]
    ).contiguous()


def assign_flat_gradient(model, flat):
    offset = 0
    for parameter in model.parameters():
        size = parameter.numel()
        parameter.grad = flat[offset:offset + size].view_as(parameter).clone()
        offset += size
