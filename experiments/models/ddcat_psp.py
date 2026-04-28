import os
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from segmentation_attacks.base import BaseGradientAttack
from torch import nn
from utils.tensor import (
    tensor_add,
    tensor_pad,
    tensor_resize,
    tensor_scale,
    tensor_select,
    zeros_like,
)

import models.backbones.resnet_ddcat as models

from .wrapper import SegmentationModelWrapper


def net_process(
    model,
    image,
    target,
    flip: bool | None = None,
    attack: BaseGradientAttack | None = None,
    attack_kwargs: dict | None = None,
):
    """
    image: Tensor of shape (C, H, W) or (H, W, C)
    mean: list of floats, one per channel
    std: list of floats, one per channel or None
    """
    if attack is None and attack_kwargs is not None and len(attack_kwargs) > 0:
        raise ValueError("attack_kwargs should be empty if attack is None")
    if attack is not None and flip:
        raise ValueError("flip and attack cannot be both True")
    flip = flip if flip is not None else attack is None
    batch_size = image.shape[0]

    if flip:
        image = torch.cat([image, image.flip(3)], dim=0)
        target = torch.cat([target, target.flip(2)], dim=0)

    if attack is not None:
        attack_results = attack.perturb(image, target, **attack_kwargs)
        if isinstance(attack_results, Sequence):
            image_adv = attack_results[0]
            state_at_iters = attack_results[1]
        else:
            image_adv = attack_results
            state_at_iters = None
        image_adv = image_adv.detach()
        output = model(image_adv)
    else:
        output = model(image)

    _, _, h_i, w_i = image.shape
    _, _, h_o, w_o = output.shape

    if h_o != h_i or w_o != w_i:
        output = F.interpolate(
            output, size=(h_i, w_i), mode="bilinear", align_corners=True
        )

    output = F.softmax(output, dim=1)
    if flip:
        return (output[:batch_size] + output[batch_size:].flip(3)) / 2
    else:
        if attack is not None:
            if state_at_iters is not None:
                return output, state_at_iters
            else:
                return output
        return output


def scale_process_cv2(
    model,
    image,
    target,
    classes,
    crop_h,
    crop_w,
    h,
    w,
    mean,
    ignore_index,
    stride_rate=2 / 3,
    flip: bool | None = False,
    attack: BaseGradientAttack | None = None,
):
    # original paper code
    ori_h, ori_w, _ = image.shape
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(
            image,
            pad_h_half,
            pad_h - pad_h_half,
            pad_w_half,
            pad_w - pad_w_half,
            cv2.BORDER_CONSTANT,
            value=mean,
        )
        target = cv2.copyMakeBorder(
            target,
            pad_h_half,
            pad_h - pad_h_half,
            pad_w_half,
            pad_w - pad_w_half,
            cv2.BORDER_CONSTANT,
            value=ignore_index,  # ignore index
        )

    new_h, new_w, _ = image.shape
    stride_h = int(np.ceil(crop_h * stride_rate))
    stride_w = int(np.ceil(crop_w * stride_rate))
    grid_h = int(np.ceil(float(new_h - crop_h) / stride_h) + 1)
    grid_w = int(np.ceil(float(new_w - crop_w) / stride_w) + 1)
    prediction_crop = np.zeros((new_h, new_w, classes), dtype=float)
    count_crop = np.zeros((new_h, new_w), dtype=float)
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[s_h:e_h, s_w:e_w].copy()

            target_crop = target[s_h:e_h, s_w:e_w].copy()
            count_crop[s_h:e_h, s_w:e_w] += 1
            image_crop = torch.from_numpy(image_crop.transpose((2, 0, 1))).float()
            image_crop = image_crop.unsqueeze(0).to(model.device)
            target_crop = torch.from_numpy(target_crop).long()
            target_crop = target_crop.unsqueeze(0).to(model.device)
            model_output = net_process(
                model, image_crop, target_crop, flip=flip, attack=attack
            ).squeeze()
            model_output = model_output.data.cpu().numpy()
            model_output = model_output.transpose(1, 2, 0)
            prediction_crop[s_h:e_h, s_w:e_w, :] += model_output
    prediction_crop /= np.expand_dims(count_crop, 2)
    prediction_crop = prediction_crop[
        pad_h_half : pad_h_half + ori_h, pad_w_half : pad_w_half + ori_w
    ]
    prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return prediction


def scale_process_pt(
    model,
    image,
    target,
    classes,
    crop_h,
    crop_w,
    h,
    w,
    mean,
    ignore_index,
    stride_rate=2 / 3,
    flip: bool | None = False,
    attack: BaseGradientAttack | None = None,
    attack_kwargs: dict | None = None,
):
    # Get original dimensions
    batch_size, channels, ori_h, ori_w = image.shape

    # Calculate padding
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)

    # Perform padding if needed
    if pad_h > 0 or pad_w > 0:
        # Convert mean to appropriate tensor format
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean, device=model.device)

        # Padding for image
        padding = (pad_w_half, pad_w - pad_w_half, pad_h_half, pad_h - pad_h_half)
        # image = torch.nn.functional.pad(image, padding, mode="constant", value=mean)
        image = tensor_pad(image, padding, mode="constant", value=mean)

        # Padding for target
        if target is not None:
            if len(target.shape) == 2:  # HW format
                target = target.unsqueeze(0).unsqueeze(0)  # Convert to NCHW
            target = torch.nn.functional.pad(
                target, padding, mode="constant", value=ignore_index
            )

    # Get new dimensions after padding
    _, _, new_h, new_w = image.shape

    # Calculate stride and grid parameters
    stride_h = int(np.ceil(crop_h * stride_rate))
    stride_w = int(np.ceil(crop_w * stride_rate))
    grid_h = int(np.ceil(float(new_h - crop_h) / stride_h) + 1)
    grid_w = int(np.ceil(float(new_w - crop_w) / stride_w) + 1)

    # Initialize prediction and count tensors
    merged_outputs = None
    count_crop = torch.zeros((batch_size, 1, new_h, new_w), device=model.device)

    # Process each crop
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w

            # Extract crop
            image_crop = image[:, :, s_h:e_h, s_w:e_w].clone()
            target_crop = target[:, s_h:e_h, s_w:e_w].clone()

            # Update count
            count_crop[:, :, s_h:e_h, s_w:e_w] += 1

            # Forward pass
            crop_outputs = net_process(
                model,
                image_crop,
                target_crop,
                flip=flip,
                attack=attack,
                attack_kwargs=attack_kwargs,
            )
            if merged_outputs is None:
                merged_outputs = zeros_like(
                    crop_outputs, (batch_size, None, new_h, new_w)
                )
            # Add to prediction
            tensor_add(
                merged_outputs,
                crop_outputs,
                (slice(None), slice(None), slice(s_h, e_h), slice(s_w, e_w)),
            )

    # Average predictions by count
    tensor_scale(
        merged_outputs,
        1.0 / count_crop,
        (slice(None), slice(None), slice(None), slice(None)),
    )

    # Crop to original size
    merged_outputs = tensor_select(
        merged_outputs,
        (
            slice(None),
            slice(None),
            slice(pad_h_half, pad_h_half + ori_h),
            slice(pad_w_half, pad_w_half + ori_w),
        ),
    )
    merged_outputs = tensor_resize(merged_outputs, (h, w))
    return merged_outputs


class DDCATModel(SegmentationModelWrapper):
    num_classes_attr = "num_class"

    def __init__(self, **kwargs):
        super().__init__(
            num_classes_attr=self.num_classes_attr,
            checkpoint_key="state_dict",
            checkpoint_prefix="module.",
            **kwargs,
        )


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins, BatchNorm):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin),
                    nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                    BatchNorm(reduction_dim),
                    nn.ReLU(inplace=True),
                )
            )
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(
                F.interpolate(f(x), x_size[2:], mode="bilinear", align_corners=True)
            )
        return torch.cat(out, 1)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, BatchNorm):
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                BatchNorm(out_channels),
                nn.ReLU(),
            )
        )
        for atrous_rate in atrous_rates:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        3,
                        padding=atrous_rate,
                        dilation=atrous_rate,
                        bias=False,
                    ),
                    BatchNorm(out_channels),
                    nn.ReLU(),
                )
            )
        modules.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                BatchNorm(out_channels),
                nn.ReLU(),
            )
        )
        self.convs = nn.ModuleList(modules)

    def forward(self, x):
        res = []
        for conv in self.convs[:-1]:
            res.append(conv(x))
        res.append(
            F.interpolate(
                self.convs[-1](x),
                x.shape[2:],
                mode="bilinear",
                align_corners=True,
            )
        )
        return torch.cat(res, dim=1)


class DeepLabV3(nn.Module):
    def __init__(
        self,
        layers=50,
        atrous_rates=(6, 12, 18),
        dropout=0.1,
        classes=2,
        zoom_factor=8,
        use_aspp=True,
        criterion=nn.CrossEntropyLoss(ignore_index=255),
        BatchNorm=nn.BatchNorm2d,
        pretrained=True,
    ):
        super(DeepLabV3, self).__init__()
        assert layers in [50, 101, 152]
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_aspp = use_aspp
        self.criterion = criterion
        models.BatchNorm = BatchNorm

        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu1,
            resnet.conv2,
            resnet.bn2,
            resnet.relu2,
            resnet.conv3,
            resnet.bn3,
            resnet.relu3,
            resnet.maxpool,
        )
        self.layer1, self.layer2, self.layer3, self.layer4 = (
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        for n, m in self.layer3.named_modules():
            if "conv2" in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif "downsample.0" in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if "conv2" in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif "downsample.0" in n:
                m.stride = (1, 1)

        in_channels, out_channels = 2048, 256
        if use_aspp:
            self.aspp = ASPP(in_channels, out_channels, atrous_rates, BatchNorm)
            fea_dim = out_channels * (len(atrous_rates) + 2)

        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, out_channels, kernel_size=1, bias=False),
            BatchNorm(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_channels, classes, kernel_size=1),
        )

        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(in_channels // 2, out_channels, kernel_size=1, bias=False),
                BatchNorm(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(out_channels, classes, kernel_size=1),
            )

    def forward(self, x, y=None, indicate=0):
        x_size = x.size()
        assert (x_size[2] - 1) % 8 == 0 and (x_size[3] - 1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)
        if self.use_aspp:
            x = self.aspp(x)
        x = self.cls(x)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)

        if self.training or indicate == 1:
            aux = self.aux(x_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(
                    aux, size=(h, w), mode="bilinear", align_corners=True
                )
            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)
            return x.max(1)[1], main_loss, aux_loss, x
        else:
            return x


class DeepLabV3_DDCAT(nn.Module):
    def __init__(
        self,
        layers=50,
        atrous_rates=(6, 12, 18),
        dropout=0.1,
        classes=2,
        zoom_factor=8,
        use_aspp=True,
        criterion=nn.CrossEntropyLoss(ignore_index=255),
        BatchNorm=nn.BatchNorm2d,
        pretrained=True,
    ):
        super(DeepLabV3_DDCAT, self).__init__()
        assert layers in [50, 101, 152]
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_aspp = use_aspp
        self.criterion = criterion
        models.BatchNorm = BatchNorm
        self.criterion_mask = nn.CrossEntropyLoss()

        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu1,
            resnet.conv2,
            resnet.bn2,
            resnet.relu2,
            resnet.conv3,
            resnet.bn3,
            resnet.relu3,
            resnet.maxpool,
        )
        self.layer1, self.layer2, self.layer3, self.layer4 = (
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        for n, m in self.layer3.named_modules():
            if "conv2" in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif "downsample.0" in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if "conv2" in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif "downsample.0" in n:
                m.stride = (1, 1)

        in_channels, out_channels = 2048, 256
        if use_aspp:
            self.aspp = ASPP(in_channels, out_channels, atrous_rates, BatchNorm)
            fea_dim = out_channels * (len(atrous_rates) + 2)

        self.cls1 = nn.Sequential(
            nn.Conv2d(fea_dim, out_channels, kernel_size=1, bias=False),
            BatchNorm(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_channels, classes, kernel_size=1),
        )
        ################################################################
        self.mask1 = nn.Sequential(
            nn.Conv2d(fea_dim, out_channels, kernel_size=1, bias=False),
            BatchNorm(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_channels, 2, kernel_size=1),
        )
        ################################################################
        self.cls2 = nn.Sequential(
            nn.Conv2d(fea_dim, out_channels, kernel_size=1, bias=False),
            BatchNorm(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_channels, classes, kernel_size=1),
        )
        ################################################################

        if self.training:
            self.aux_cls1 = nn.Sequential(
                nn.Conv2d(in_channels // 2, out_channels, kernel_size=1, bias=False),
                BatchNorm(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(out_channels, classes, kernel_size=1),
            )

    def forward(self, x, y_target=None, indicate_map=None, indicate=0):
        x_size = x.size()
        assert (x_size[2] - 1) % 8 == 0 and (x_size[3] - 1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x_tmp2 = self.layer4(x_tmp)
        if self.use_aspp:
            x = self.aspp(x_tmp2)
        result_normal = self.cls1(x)
        result_adver = self.cls2(x)
        mask1 = self.mask1(x)
        if self.zoom_factor != 1:
            result_normal = F.interpolate(
                result_normal, size=(h, w), mode="bilinear", align_corners=True
            )
            result_adver = F.interpolate(
                result_adver, size=(h, w), mode="bilinear", align_corners=True
            )
            mask1 = F.interpolate(
                mask1, size=(h, w), mode="bilinear", align_corners=True
            )

        if self.training or indicate == 1:
            indiatce_adver = mask1.max(1)[1]
            indiatce_adver = indiatce_adver.unsqueeze(1)
            indiatce_adver = indiatce_adver.expand_as(result_adver)
            indiatce_normal = 1 - indiatce_adver
            indiatce_adver = indiatce_adver.float()
            indiatce_normal = indiatce_normal.float()
            final_result = (
                indiatce_adver * result_adver + indiatce_normal * result_normal
            )

            aux_cls_normal = self.aux_cls1(x_tmp)
            if self.zoom_factor != 1:
                aux_cls_normal = F.interpolate(
                    aux_cls_normal,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=True,
                )
            aux_final_results = aux_cls_normal

            if y_target is None:
                temp_y = (
                    torch.zeros(
                        [
                            result_normal.shape[0],
                            result_normal.shape[2],
                            result_normal.shape[3],
                        ]
                    )
                    .cuda()
                    .long()
                )
                main_loss = self.criterion(final_result, temp_y)
                aux_loss = self.criterion(aux_final_results, temp_y)
            else:
                main_loss = self.criterion(final_result, y_target)
                aux_loss = self.criterion(aux_final_results, y_target)

            if indicate_map is None:
                temp_map = (
                    torch.zeros([mask1.shape[0], mask1.shape[2], mask1.shape[3]])
                    .cuda()
                    .long()
                )
                main_loss += self.criterion_mask(mask1, temp_map)
            else:
                main_loss += self.criterion_mask(mask1, indicate_map)

            return (
                final_result.max(1)[1],
                main_loss,
                aux_loss,
                final_result,
                result_normal,
            )
        else:
            final_result = result_normal
            return final_result


class PSPNet(nn.Module):
    def __init__(
        self,
        layers=50,
        classes=21,
        bins=(1, 2, 3, 6),
        dropout=0.1,
        zoom_factor=8,
        use_ppm=True,
        criterion=nn.CrossEntropyLoss(ignore_index=-1),
        BatchNorm=nn.BatchNorm2d,
        pretrained=True,
        clean=True,
    ):
        super(PSPNet, self).__init__()
        assert layers in [50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion
        models.BatchNorm = BatchNorm
        self.num_class = classes

        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)

        if not clean:
            self.layer0 = nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu1, resnet.maxpool
            )
        else:
            self.layer0 = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu1,
                resnet.conv2,
                resnet.bn2,
                resnet.relu2,
                resnet.conv3,
                resnet.bn3,
                resnet.relu3,
                resnet.maxpool,
            )

        self.layer1, self.layer2, self.layer3, self.layer4 = (
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        for n, m in self.layer3.named_modules():
            if "conv2" in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif "downsample.0" in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if "conv2" in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif "downsample.0" in n:
                m.stride = (1, 1)

        fea_dim = 2048
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim / len(bins)), bins, BatchNorm)
            fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            BatchNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1),
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                BatchNorm(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1),
            )

    def forward(self, x, y=None, indicate=0):
        x_size = x.size()
        assert (x_size[2] - 1) % 8 == 0 and (x_size[3] - 1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)
        if self.use_ppm:
            x = self.ppm(x)
        x = self.cls(x)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)

        if self.training or indicate == 1:
            aux = self.aux(x_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(
                    aux, size=(h, w), mode="bilinear", align_corners=True
                )
            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)
            return main_loss, aux_loss, x
        else:
            return x


class PSPNet_DDCAT(nn.Module):
    def __init__(
        self,
        layers=50,
        bins=(1, 2, 3, 6),
        dropout=0.1,
        classes=2,
        zoom_factor=8,
        use_ppm=True,
        criterion=nn.CrossEntropyLoss(ignore_index=255),
        BatchNorm=nn.BatchNorm2d,
        pretrained=False,
    ):
        super(PSPNet_DDCAT, self).__init__()
        assert layers in [50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion
        models.BatchNorm = BatchNorm
        self.num_class = classes
        self.criterion_mask = nn.CrossEntropyLoss()

        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)

        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu1,
            resnet.conv2,
            resnet.bn2,
            resnet.relu2,
            resnet.conv3,
            resnet.bn3,
            resnet.relu3,
            resnet.maxpool,
        )
        self.layer1, self.layer2, self.layer3, self.layer4 = (
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        for n, m in self.layer3.named_modules():
            if "conv2" in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif "downsample.0" in n:
                m.stride = (1, 1)

        for n, m in self.layer4.named_modules():
            if "conv2" in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif "downsample.0" in n:
                m.stride = (1, 1)

        fea_dim = 2048
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim / len(bins)), bins, BatchNorm)
            fea_dim *= 2

        ################################################################
        self.cls1 = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            BatchNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1),
        )
        ################################################################
        self.mask1 = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            BatchNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, 2, kernel_size=1),
        )
        ################################################################
        self.cls2 = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            BatchNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1),
        )
        ################################################################
        if self.training:
            self.aux_cls1 = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                BatchNorm(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1),
            )

    def forward(self, x, y_target=None, indicate_map=None, indicate=0):
        x_size = x.size()
        assert (x_size[2] - 1) % 8 == 0 and (x_size[3] - 1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x_tmp2 = self.layer4(x_tmp)
        if self.use_ppm:
            x_ppm = self.ppm(x_tmp2)

        result_normal = self.cls1(x_ppm)
        result_adver = self.cls2(x_ppm)
        mask1 = self.mask1(x_ppm)
        if self.zoom_factor != 1:
            result_normal = F.interpolate(
                result_normal, size=(h, w), mode="bilinear", align_corners=True
            )
            result_adver = F.interpolate(
                result_adver, size=(h, w), mode="bilinear", align_corners=True
            )
            mask1 = F.interpolate(
                mask1, size=(h, w), mode="bilinear", align_corners=True
            )

        if self.training or indicate == 1:
            indiatce_adver = mask1.max(1)[1]
            indiatce_adver = indiatce_adver.unsqueeze(1)
            indiatce_adver = indiatce_adver.expand_as(result_adver)
            indiatce_normal = 1 - indiatce_adver
            indiatce_adver = indiatce_adver.float()
            indiatce_normal = indiatce_normal.float()
            final_result = (
                indiatce_adver * result_adver + indiatce_normal * result_normal
            )

            aux_cls_normal = self.aux_cls1(x_tmp)
            if self.zoom_factor != 1:
                aux_cls_normal = F.interpolate(
                    aux_cls_normal,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=True,
                )
            aux_final_results = aux_cls_normal

            if y_target is None:
                temp_y = (
                    torch.zeros(
                        [
                            result_normal.shape[0],
                            result_normal.shape[2],
                            result_normal.shape[3],
                        ]
                    )
                    .cuda()
                    .long()
                )
                main_loss = self.criterion(final_result, temp_y)
                aux_loss = self.criterion(aux_final_results, temp_y)
            else:
                main_loss = self.criterion(final_result, y_target)
                aux_loss = self.criterion(aux_final_results, y_target)

            if indicate_map is None:
                temp_map = (
                    torch.zeros([mask1.shape[0], mask1.shape[2], mask1.shape[3]])
                    .cuda()
                    .long()
                )
                main_loss += self.criterion_mask(mask1, temp_map)
            else:
                main_loss += self.criterion_mask(mask1, indicate_map)

            return (
                final_result.max(1)[1],
                main_loss,
                aux_loss,
                final_result,
                result_normal,
            )
        else:
            final_result = result_normal
            return final_result


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    input = torch.rand(4, 3, 473, 473).cuda()
    model = PSPNet(
        layers=50,
        bins=(1, 2, 3, 6),
        dropout=0.1,
        classes=21,
        zoom_factor=1,
        use_ppm=True,
        pretrained=True,
    ).cuda()
    model.eval()
    print(model)
    output = model(input)
    print("PSPNet", output.size())
