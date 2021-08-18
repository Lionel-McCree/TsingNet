import math
import torch
import cv2
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils_box.anchors import gen_anchors, box_iou
from libs.sigmoid_focal_loss import sigmoid_focal_loss
from libs.nms import box_nms
# from backbone import resnet101 as backbone
from backbone import resnet50 as backbone
from rfb_block import RFBblock


class Detector(nn.Module):
    def __init__(self, pretrained=False):
        super(Detector,self).__init__()

        # anchor的宽高设置
        self.a_hw = [
            [15.8, 17.5],
            [20.4, 22.75],
            [24.4, 31.25],

            [30.6, 35.5],
            [36.6, 39.25],
            [46, 48.75],

            [57.8, 62.25],
            [77.2, 83.25],
            [115.4, 121.5]
        ]
        # p3 p4 p5 p6 p7
        self.scales = 5
        self.first_stride = 8
        self.train_size = 1024
        self.eval_size = 1024
        self.iou_th = (0.4, 0.5)
        self.classes = 45
        self.nms_th = 0.05
        self.nms_iou = 0.5
        self.max_detections = 3000
        self.scale_loss = torch.nn.MSELoss()

        #-----------------------
        self.rfb = RFBblock(residual=True)
        self.backbone = backbone(pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)
        self.conv_out6 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)
        self.conv_out7 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)

        self.prj_5 = nn.Conv2d(2048, 256, kernel_size=1)
        self.prj_4 = nn.Conv2d(1024, 256, kernel_size=1)
        self.prj_3 = nn.Conv2d(512, 256, kernel_size=1)

        # self.conv_5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # self.conv_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # self.conv_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # 空洞卷积
        self.down_conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, dilation=1)
        self.down_conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=2, dilation=2)
        self.down_conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=3, stride=1, dilation=3)

        self.down = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)

        # 通道全局平均池化 senet
        self.global_channel = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_features=256, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=256)
        self.sigmoid = nn.Sigmoid()
        # 空间注意力网络
        # self.conv1 = nn.Conv2d(256, 256, kernel_size=1)
        # self.bn = nn.BatchNorm2d(256)
        # self.sigmoid = nn.Sigmoid()

        self.conv_cls = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, len(self.a_hw)*self.classes, kernel_size=3, padding=1)
        )

        self.conv_neg = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, len(self.a_hw)*4, kernel_size=3, padding=1)
        )
        # 对上述卷积层进行初始化
        #children() 返回当前模型的所有层
        for layer in self.conv_cls.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.constant_(layer.bias, 0)
                nn.init.normal_(layer.weight, mean=0, std=0.01)
        for layer in self.conv_neg.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.constant_(layer.bias, 0)
                nn.init.normal_(layer.weight, mean=0, std=0.01)

        pi = 0.01
        _bias = -math.log((1.0-pi)/pi)
        nn.init.constant_(self.conv_cls[-1].bias, _bias)

        # 产生先验框==============================
        self._train_anchors_yxyx, self._train_anchors_yxhw = gen_anchors(
            self.a_hw, self.scales, self.train_size, self.first_stride
        )
        self._eval_anchors_yxyx, self._eval_anchors_yxhw = gen_anchors(
            self.a_hw, self.scales, self.eval_size, self.first_stride
        )
        self.train_an = self._train_anchors_yxyx.shape[0]

        self.register_buffer('train_anchors_yxyx', self._train_anchors_yxyx)
        self.register_buffer('eval_anchors_yxyx', self._eval_anchors_yxyx)
        self.register_buffer('train_anchors_hw', self._train_anchors_yxhw[:, 2:])
        self.register_buffer('eval_anchors_hw', self._eval_anchors_yxhw[:, 2:])

    def upsample(self, input):
        return F.interpolate(input, size=(input.shape[2]*2, input.shape[3]*2),
                             mode='bilinear', align_corners=True)

    def upsample4(self, input):
        return F.interpolate(input, size=(input.shape[2] * 4, input.shape[3] * 4),
                             mode='bilinear', align_corners=True)

    # 添加SE模块部分
    def channel_weight(self, input):
        channel_feature = self.global_channel(input)
        channel_feature = channel_feature.view(channel_feature.size(0), -1)
        out = self.fc1(channel_feature)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        out = input * out
        return out

    def forward(self, x, label_class=None, label_box=None):
        C3, C4, C5 = self.backbone(x)

        P5 = self.prj_5(C5)
        P4 = self.prj_4(C4)
        P3 = self.prj_3(C3)

        # 聚合P5 自底向上的聚合
        P3d = self.down_conv3(P3)

        P4_du = P3d + P4

        P4d = self.down_conv4(P4_du)

        P5d = self.down_conv5(P5)

        P5_du = P4d + P5d
        # 后续的网络senet 或者 SAU
        P5_a = self.channel_weight(P5_du)
        P5_au = self.upsample(P5_a)

        P4_fu = P5_au + P4
        #
        P4_a = self.channel_weight(P4_fu)
        P4_au = self.upsample(P4_a)

        P3_fu = P4_au + P3

        # 第二个
        # RFB模块
        P3 = self.rfb(P3_fu)
        P4 = self.rfb(P4_fu)
        P5 = self.rfb(P5_du)

        # p3->p3_mean
        M5 = self.upsample4(P5)
        M4 = self.upsample(P4)
        P3_m = (M5 + M4 + P3) / 3

        # P4->p4_mean
        M5_2 = self.upsample(P5)
        M3 = self.down(P3)
        P4_m = (M5_2 + P4 + M3) / 3

        P6 = self.conv_out6(P5_du)
        P7 = self.conv_out7(self.relu(P6))

        # P5->p5_mean
        P4_2 = self.down(P4)
        M6 = self.upsample(P6)
        P5_m = (P4_2 + P5  + M6) / 3


        pred_list = [P3, P4, P5, P6, P7]

        cls_out = []
        reg_out = []
        for item in pred_list:
            #---------------
            cls_i = self.conv_cls(item)
            reg_i = self.conv_neg(item)
            # cls_i[batch, an*classes, H, W] -> [batch, H*W*an, classes]
            cls_i = cls_i.permute(0, 2, 3, 1).contiguous()
            cls_i = cls_i.view(cls_i.shape[0], -1, self.classes)
            # reg_i [batch, an*4,H, W] -> [batch, H*W*an, 4]
            reg_i = reg_i.permute(0, 2, 3, 1).contiguous()
            reg_i = reg_i.view(reg_i.shape[0], -1, 4)
            cls_out.append(cls_i)
            reg_out.append(reg_i)
        cls_out = torch.cat(cls_out, dim=1)
        reg_out = torch.cat(reg_out, dim=1)
        if (label_class is None) or (label_box is None):
            return self._decode(cls_out, reg_out)
        else:
            targets_cls, targets_reg = self._encode(label_class, label_box)
            mask_cls = targets_cls > -1
            mask_reg = targets_cls> 0
            num_pos = torch.sum(mask_reg, dim=1).clamp_(min=1)
            loss = []
            # ???
            for b in range(targets_cls.shape[0]):
                cls_out_b = cls_out[b][mask_cls[b]]  # (S+-, classes)
                targets_cls_b = targets_cls[b][mask_cls[b]]  # (S+-)
                reg_out_b = reg_out[b][mask_reg[b]]  # (S+, 4)
                targets_reg_b = targets_reg[b][mask_reg[b]]  # # (S+, 4)
                loss_cls_b = sigmoid_focal_loss(cls_out_b, targets_cls_b, 2.0, 0.25).sum().view(1)
                loss_reg_b = F.smooth_l1_loss(reg_out_b, targets_reg_b, reduction='sum').view(1)
                loss.append((loss_cls_b + loss_reg_b) / float(num_pos[b]))
            loss_3 = self.scale_loss(P3, P3_m)
            loss_4 = self.scale_loss(P4, P4_m)
            loss_5 = self.scale_loss(P5, P5_m)
            loss_scale = loss_3 + loss_4 + loss_5
            return torch.cat(loss, dim=0), loss_scale  # (b)

    def _encode(self, label_class, label_box):
        label_class_out = []
        label_box_out = []
        for b in range(label_class.shape[0]):
            label_class_out_b = torch.full((self.train_an,), -1).long().to(label_class.device)
            label_box_out_b = torch.zeros(self.train_an, 4).to(label_class.device)
            iou = box_iou(self.train_anchors_yxyx, label_box[b])
            if (iou.shape[1] <= 0):
                label_class_out_b[:] = 0
                label_class_out.append(label_class_out_b)
                label_box_out.append(label_box_out_b)
                continue
            iou_max, iou_max_idx = torch.max(iou, dim=1)
            anchors_pos_mask = iou_max > self.iou_th[1]
            anchors_neg_mask = iou_max < self.iou_th[0]
            label_class_out_b[anchors_neg_mask] = 0

            label_select = iou_max_idx[anchors_pos_mask]
            label_class_out_b[anchors_pos_mask] = label_class[b][label_select]

            #
            lb_yxyx = label_box[b][label_select]
            d_yxyx = lb_yxyx - self.train_anchors_yxyx[anchors_pos_mask]
            anchors_hw = self.train_anchors_hw[anchors_pos_mask]
            d_yxyx[:, :2] = d_yxyx[:, :2]/anchors_hw/0.2
            d_yxyx[:, 2:] = d_yxyx[:, 2:]/anchors_hw/0.2
            label_box_out_b[anchors_pos_mask] = d_yxyx
            label_class_out.append(label_class_out_b)
            label_box_out.append(label_box_out_b)
        targets_cls = torch.stack(label_class_out, dim=0)
        targets_reg = torch.stack(label_box_out, dim=0)
        return targets_cls, targets_reg

    def _decode(self, cls_out, reg_out):
        cls_p_preds, cls_i_preds = torch.max(cls_out.sigmoid(), dim=2)
        cls_i_preds = cls_i_preds + 1
        reg_preds = []
        for b in range(cls_out.shape[0]):
            reg_dyxyx = reg_out[b]
            reg_dyxyx[:, :2] = reg_dyxyx[:, :2] * 0.2 * self.eval_anchors_hw
            reg_dyxyx[:, 2:] = reg_dyxyx[:, 2:] * 0.2 * self.eval_anchors_hw
            reg_yxyx = reg_dyxyx + self.eval_anchors_yxyx
            reg_preds.append(reg_yxyx)
        reg_preds = torch.stack(reg_preds, dim=0)

        # Topk
        nms_maxnum = min(int(self.max_detections), int(cls_p_preds.shape[1]))
        select = torch.topk(cls_p_preds, nms_maxnum, largest=True, dim=1)[1]
        list_cls_i_preds = []
        list_cls_p_preds = []
        list_reg_preds = []
        for b in range(cls_out.shape[0]):
            cls_i_preds_b = cls_i_preds[b][select[b]]  # (topk)
            cls_p_preds_b = cls_p_preds[b][select[b]]  # (topk)
            reg_preds_b = reg_preds[b][select[b]]  # (topk, 4)
            list_cls_i_preds.append(cls_i_preds_b)
            list_cls_p_preds.append(cls_p_preds_b)
            list_reg_preds.append(reg_preds_b)
        return torch.stack(list_cls_i_preds, dim=0), torch.stack(list_cls_p_preds, dim=0), torch.stack(list_reg_preds, dim=0)


def get_loss(temp):
    return torch.mean(temp)


def get_pred(temp, nms_th, nms_iou, eval_size):
    cls_i_preds, cls_p_preds, reg_preds = temp

    list_cls_i_preds = []
    list_cls_p_preds = []
    list_reg_preds = []

    for b in range(cls_i_preds.shape[0]):

        cls_i_preds_b = cls_i_preds[b]
        cls_p_preds_b = cls_p_preds[b]
        reg_preds_b = reg_preds[b]
        
        mask = cls_p_preds_b > nms_th
        cls_i_preds_b = cls_i_preds_b[mask]
        cls_p_preds_b = cls_p_preds_b[mask]
        reg_preds_b = reg_preds_b[mask]

        keep = box_nms(reg_preds_b, cls_p_preds_b, nms_iou)
        cls_i_preds_b = cls_i_preds_b[keep]
        cls_p_preds_b = cls_p_preds_b[keep]
        reg_preds_b = reg_preds_b[keep]

        reg_preds_b[:, :2] = reg_preds_b[:, :2].clamp(min=0)
        reg_preds_b[:, 2:] = reg_preds_b[:, 2:].clamp(max=eval_size-1)

        list_cls_i_preds.append(cls_i_preds_b)
        list_cls_p_preds.append(cls_p_preds_b)
        list_reg_preds.append(reg_preds_b)
    
    return list_cls_i_preds, list_cls_p_preds, list_reg_preds