from torch import Tensor, nn


class PSAP(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        ch = c2 // 2
        self.conv_q_right = nn.Conv2d(c1, 1, 1, bias=False)
        self.conv_v_right = nn.Conv2d(c1, ch, 1, bias=False)
        self.conv_up = nn.Conv2d(ch, c2, 1, bias=False)

        self.conv_q_left = nn.Conv2d(c1, ch, 1, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(c1, ch, 1, bias=False)

    def spatial_pool(self, x: Tensor) -> Tensor:
        input_x = self.conv_v_right(x)  # [B, C, H, W]
        context_mask = self.conv_q_right(x)  # [B, 1, H, W]
        B, C, _, _ = input_x.shape

        input_x = input_x.view(B, C, -1)
        context_mask = context_mask.view(B, 1, -1).softmax(dim=2)

        context = input_x @ context_mask.transpose(1, 2)
        context = self.conv_up(context.unsqueeze(-1)).sigmoid()
        x *= context
        return x

    def channel_pool(self, x: Tensor) -> Tensor:
        g_x = self.conv_q_left(x)
        B, C, H, W = g_x.shape
        avg_x = self.avg_pool(g_x).view(B, C, -1).permute(0, 2, 1)
        theta_x = self.conv_v_left(x).view(B, C, -1)

        context = avg_x @ theta_x
        context = context.softmax(dim=2).view(B, 1, H, W).sigmoid()
        x *= context
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self.spatial_pool(x) + self.channel_pool(x)


class PSAS(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        ch = c2 // 2
        self.conv_q_right = nn.Conv2d(c1, 1, 1, bias=False)
        self.conv_v_right = nn.Conv2d(c1, ch, 1, bias=False)
        self.conv_up = nn.Sequential(
            nn.Conv2d(ch, ch // 4, 1),
            nn.LayerNorm([ch // 4, 1, 1]),
            nn.ReLU(),
            nn.Conv2d(ch // 4, c2, 1),
        )

        self.conv_q_left = nn.Conv2d(c1, ch, 1, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(c1, ch, 1, bias=False)

    def spatial_pool(self, x: Tensor) -> Tensor:
        input_x = self.conv_v_right(x)  # [B, C, H, W]
        context_mask = self.conv_q_right(x)  # [B, 1, H, W]
        B, C, _, _ = input_x.shape

        input_x = input_x.view(B, C, -1)
        context_mask = context_mask.view(B, 1, -1).softmax(dim=2)

        context = input_x @ context_mask.transpose(1, 2)
        context = self.conv_up(context.unsqueeze(-1)).sigmoid()
        x *= context
        return x

    def channel_pool(self, x: Tensor) -> Tensor:
        g_x = self.conv_q_left(x)
        B, C, H, W = g_x.shape
        avg_x = self.avg_pool(g_x).view(B, C, -1).permute(0, 2, 1)
        theta_x = self.conv_v_left(x).view(B, C, -1).softmax(dim=2)

        context = avg_x @ theta_x
        context = context.view(B, 1, H, W).sigmoid()
        x *= context
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self.channel_pool(self.spatial_pool(x))
