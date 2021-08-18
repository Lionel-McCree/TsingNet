import torch
import torch.nn as nn


class RFBblock(nn.Module):
    def __init__(self, in_ch=256, residual=False):
        super(RFBblock, self).__init__()
        inter_c = in_ch // 4
        self.branch_0 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=inter_c, kernel_size=1, stride=1, padding=0)
        )
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=inter_c, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=inter_c, out_channels=inter_c, kernel_size=3, stride=1, padding=1)
        )
        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=inter_c, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=inter_c, out_channels=inter_c, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=inter_c, out_channels=inter_c, kernel_size=3, stride=1, dilation=2, padding=2),
            nn.Conv2d(in_channels=inter_c, out_channels=inter_c, kernel_size=3, stride=1, dilation=3, padding=3)
        )
        self.branch_3 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=inter_c, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=inter_c, out_channels=inter_c, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(in_channels=inter_c, out_channels=inter_c, kernel_size=3, stride=1, dilation=2, padding=2),
            nn.Conv2d(in_channels=inter_c, out_channels=inter_c, kernel_size=3, stride=1, dilation=3, padding=3)
        )
        self.residual = residual

    def forward(self, x):
        x_0 = self.branch_0(x)
        x_1 = self.branch_1(x)
        x_2 = self.branch_2(x)
        x_3 = self.branch_3(x)
        out = torch.cat((x_0, x_1, x_2, x_3), 1)
        if self.residual:
            out += x
        return out