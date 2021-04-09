import torch.nn as nn
import torch.nn.functional as F

# from mmcv.cnn import constant_init, kaiming_init, normal_init


class FBOTAM(nn.Module):
    """Temporal Adaptive Module(TAM) for FBO in LFB.

    Args:
        in_channels (int): Channel num of input features.
        num_segments (int): Number of frame segments.
        alpha (int): ```alpha``` in the paper and is the ratio of the
            intermediate channel number to the initial channel number in the
            global branch. Default: 2.
        adaptive_kernel_size (int): ```K``` in the paper and is the size of the
            adaptive kernel size in the global branch. Default: 3.
        beta (int): ```beta``` in the paper and is set to control the model
            complexity in the local branch. Default: 4.
        conv1d_kernel_size (int): Size of the convolution kernel of Conv1d in
            the local branch. Default: 3.
        adaptive_convolution_stride (int): The first dimension of strides in
            the adaptive convolution of ```Temporal Adaptive Aggregation```.
            Default: 1.
        adaptive_convolution_padding (int): The first dimension of paddings in
            the adaptive convolution of ```Temporal Adaptive Aggregation```.
            Default: 1.
        init_std (float): Std value for initiation of `nn.Linear`. Default:
            0.001.
    """

    def __init__(
            self,
            st_feat_channels,
            lt_feat_channels,
            num_st_feat,
            num_lt_feat,
            alpha=4,  # 4 * alpha for slowonly_4x16
            adaptive_kernel_size=3,
            beta=4,
            conv1d_kernel_size=3,
            adaptive_convolution_stride=1,
            adaptive_convolution_padding=1,
            init_std=0.001):
        super().__init__()

        assert beta > 0 and alpha > 0
        self.st_feat_channels = st_feat_channels
        self.lt_feat_channels = lt_feat_channels
        self.num_st_feat = num_st_feat
        self.num_lt_feat = num_lt_feat
        self.alpha = alpha
        self.adaptive_kernel_size = adaptive_kernel_size
        self.beta = beta
        self.conv1d_kernel_size = conv1d_kernel_size
        self.adaptive_convolution_stride = adaptive_convolution_stride
        self.adaptive_convolution_padding = adaptive_convolution_padding
        self.init_std = init_std

        self.G = nn.Sequential(
            nn.Linear(num_st_feat, num_st_feat * alpha, bias=False),
            nn.BatchNorm1d(num_st_feat * alpha), nn.ReLU(inplace=True),
            nn.Linear(num_st_feat * alpha, adaptive_kernel_size, bias=False),
            nn.Softmax(-1))

        self.L = nn.Sequential(
            nn.Conv1d(
                lt_feat_channels,
                lt_feat_channels // beta,
                conv1d_kernel_size,
                stride=1,
                padding=conv1d_kernel_size // 2,
                bias=False), nn.BatchNorm1d(lt_feat_channels // beta),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                lt_feat_channels // beta, lt_feat_channels, 1, bias=False),
            nn.Sigmoid())

        self.init_weights()

    def init_weights(self):
        """Initiate the parameters from scratch."""
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         kaiming_init(m)
        #     elif isinstance(m, nn.BatchNorm1d):
        #         constant_init(m, 1)
        #     elif isinstance(m, nn.Linear):
        #         normal_init(m, std=self.init_std)
        pass  # init by default

    def forward(self, st_feat, lt_feat):
        """Defines the computation performed at every call.

        Args:
            st_feat (torch.Tensor): short-term features
                [N, C, t, 1, 1]
            lt_feat (torch.Tensor): long-term features
                [N, C, window_size * num_feat_per_step, 1, 1]

        Returns:
            torch.Tensor: The output of the module.
        """
        st_n, st_c, st_t, st_h, st_w = st_feat.size()
        lt_n, lt_c, lt_t, lt_h, lt_w = lt_feat.size()

        st_theta_out = F.adaptive_avg_pool2d(
            st_feat.view(-1, st_t, st_h, st_w), (1, 1))
        conv_kernel = self.G(st_theta_out.view(st_n * st_c, st_t)).view(
            st_n * st_c, 1, -1, 1)

        lt_theta_out = F.adaptive_avg_pool2d(
            lt_feat.view(-1, lt_t, lt_h, lt_w), (1, 1))
        local_activation = self.L(lt_theta_out.view(lt_n, lt_c, lt_t)).view(
            lt_n, lt_c, lt_t, 1, 1)
        new_lt_feat = lt_feat * local_activation

        fbo_feat = F.conv2d(
            new_lt_feat.view(1, lt_n * lt_c, lt_t, lt_h * lt_w),
            conv_kernel,
            bias=None,
            stride=(self.adaptive_convolution_stride, 1),
            padding=(self.adaptive_convolution_padding, 0),
            groups=lt_n * lt_c)

        fbo_feat = fbo_feat.view(lt_n, lt_c, lt_t, lt_h, lt_w)

        return fbo_feat
