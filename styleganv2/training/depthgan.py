import numpy as np
import torch
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma
from training.networks import *
import torch.nn.functional as F


@persistence.persistent_class
class SynthesisBlockPrior(torch.nn.Module):
    '''A StyleGAN Synthesis Block with Feature Maps Injection
    Feature maps derived the prior network are concatenated with
    corresponding feature maps and convolved to retain channel numbers.
    '''
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        out_channels,                       # Number of output channels.
        w_dim,                              # Intermediate latent (W) dimensionality.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of output color channels.
        is_last,                            # Is this the last block?
        architecture        = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        **layer_kwargs,                     # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        # First Block Const Input?
        if in_channels == 0:
            self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution])) # Trainable?

        if in_channels != 0:
            self.conv_inter_0 = Conv2dLayer(in_channels=2 * in_channels, out_channels=in_channels,
                                            kernel_size=1)  # compress to in_channel for conv0
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, up=2,
                resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
            self.num_conv += 1

        self.conv_inter_1 = Conv2dLayer(in_channels=2 * out_channels, out_channels=out_channels, kernel_size=1)

        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
            conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
        self.num_conv += 1

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
                conv_clamp=conv_clamp, channels_last=self.channels_last)
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=2,
                resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, ws, prior, force_fp32=False, fused_modconv=None, **layer_kwargs):
        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            with misc.suppress_tracer_warnings(): # this value will be treated as a constant
                fused_modconv = (not self.training) and (dtype == torch.float32 or int(x.shape[0]) == 1)

        # Input.
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        else:
            misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution // 2])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
            x = torch.cat((x, prior[1]), dim=1)
            x = self.conv_inter_1(x)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = torch.cat((x,prior[0]), dim=1)
            x = self.conv_inter_0(x)
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = torch.cat((x,prior[1]), dim=1)
            x = self.conv_inter_1(x)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = y.add_(x)
        else:
            x = torch.cat((x, prior[0]), dim=1)
            x = self.conv_inter_0(x)
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = torch.cat((x, prior[1]), dim=1)
            x = self.conv_inter_1(x)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

        # ToRGB.
        if img is not None:
            misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
            img = upfirdn2d.upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img

@persistence.persistent_class
class SynthesisNetworkPrior(torch.nn.Module):
    '''Synthesis Network with depth Prior
    SynthesisBlock -> SynthesisBlockPrior
    Add Priors(List): Feature Maps from Prior Network
    '''
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output image resolution.
        img_channels,               # Number of color channels.
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        num_fp16_res    = 0,        # Use FP16 for the N highest resolutions.
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlockPrior(in_channels, out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb

            setattr(self, f'b{res}', block)

    def forward(self, ws, priors, **block_kwargs):
        block_ws = []
        assert len(priors) == len(self.block_resolutions)
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None
        for res, cur_ws, cur_prior in zip(self.block_resolutions, block_ws, priors):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, cur_prior, **block_kwargs)
        return img

@persistence.persistent_class
class SignalEncode(torch.nn.Module):
    def __init__(self,
        frequency,     # signal frequency, output channel = 2t
        w_dim,         # w latent code dimension
    ):
        super().__init__()
        self.w_dim = w_dim
        self.frequency = frequency
        self.FC_1 = FullyConnectedLayer(in_features=2*frequency, out_features=w_dim)
        self.FC_2 = FullyConnectedLayer(in_features=w_dim, out_features=w_dim)
        self.multiplier = torch.arange(1,frequency+1)

    def forward(self,
                angle,
                w_d       # Original w_d transformed from z_d through MappingNetwork
        ):

        # Frequency Sampling
        angles = self.multiplier * angle
        sin_angles = angles.sin()
        cos_angles = angles.cos()
        angles_encode = torch.cat((sin_angles,cos_angles))

        # 2-layer FC: 2t -> w_dim
        angles_encode=angles_encode.unsqueeze(0)
        angles_inter = self.FC_1(angles_encode)
        w_angle = self.FC_2(angles_inter)
        w_angle_d = w_angle * w_d

        return w_angle_d

class SynthesisNetworkDepth(torch.nn.Module):
    '''Synthesis Network with depth Prior
    Use w'(angle signal injected in the first Synthesis Block
    Use w(original) in the other Synthesis Blocks
    '''
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output image resolution.
        img_channels,               # Number of color channels.
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        num_fp16_res    = 0,        # Use FP16 for the N highest resolutions.
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
            # self.num_ws += block.num_conv
            self.num_ws += block.num_conv + block.num_torgb
            # if is_last:
                # self.num_ws += block.num_torgb

            setattr(self, f'b{res}', block)

    def forward(self, ws, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                # w_idx += block.num_conv
                w_idx += block.num_conv + block.num_torgb

        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)
        return img


@persistence.persistent_class
class GeneratorRGB(torch.nn.Module):
    '''Generator with depth prior

    '''
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        synthesis_kwargs    = {},   # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetworkPrior(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, z, c, priors, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        img = self.synthesis(ws, priors, **synthesis_kwargs)
        return img

@persistence.persistent_class
class GeneratorDepth(torch.nn.Module):
    '''Depth Map Generator
    Add Signal Encode to produce w'
    Use w' in the first Synthesis Block
    Use w in the other Synthesis Block
    '''
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        frequency,                  # Frequency of Signal Encode
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        synthesis_kwargs    = {},   # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetworkDepth(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)
        self.signalencode = SignalEncode(frequency=frequency,w_dim=w_dim)
        # Use wsa in the first two layers
        self.num_wsa = self.synthesis.b4.num_torgb + self.synthesis.b4.num_conv + self.synthesis.b8.num_torgb + self.synthesis.b8.num_conv

    def forward(self, z, c, angle, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        ws_single = ws[:,0,:]
        wsa = self.signalencode(angle,ws_single)
        # Use wsa in the first two layers
        wsa = wsa.unsqueeze(1).repeat([1, self.num_wsa, 1])
        ws[:,0:self.num_wsa,:] = wsa
        img = self.synthesis(ws, **synthesis_kwargs)
        return img

@persistence.persistent_class
class DiscriminatorBlockDepth(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        tmp_channels,                       # Number of intermediate channels.
        out_channels,                       # Number of output channels.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of input color channels.
        first_layer_idx,                    # Index of the first layer.
        architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet'.
        activation          = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        freeze_layers       = 0,            # Freeze-D: Number of layers to freeze.
    ):
        assert in_channels in [0, tmp_channels]
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        self.num_layers = 0
        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable
        trainable_iter = trainable_gen()

        if in_channels == 0: # or architecture == 'skip': No skip architecture
            # For Depth Input
            self.fromdepth = Conv2dLayer(1, tmp_channels, kernel_size=1, activation=activation,
                trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

            self.fromrgb = Conv2dLayer(img_channels, tmp_channels, kernel_size=1, activation=activation,
                trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv0 = Conv2dLayer(tmp_channels, tmp_channels, kernel_size=3, activation=activation,
            trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv1 = Conv2dLayer(tmp_channels, out_channels, kernel_size=3, activation=activation, down=2,
            trainable=next(trainable_iter), resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last)

        if architecture == 'resnet':
            self.skip = Conv2dLayer(tmp_channels, out_channels, kernel_size=1, bias=False, down=2,
                trainable=next(trainable_iter), resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, force_fp32=False):
        assert img.shape[1] in [self.img_channels, self.img_channels+1]
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        if x is not None:
            misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.No skip architecture
        if self.in_channels == 0:
            misc.assert_shape(img, [None, None, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            if img.shape[1] == self.img_channels: # RGB Input
                y = self.fromrgb(img)
                x = x + y if x is not None else y
                # img = upfirdn2d.downsample2d(img, self.resample_filter) if self.architecture == 'skip' else None
            else: # RGBD Input
                img_tuple = img.split(3,dim=1)
                img_rgb = img_tuple[0]
                img_d = img_tuple[1]
                y_rgb =self.fromrgb(img_rgb)
                y_d = self.fromdepth(img_d)
                x = y_rgb + y_d


        # Main layers.
        if self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x)
            x = self.conv1(x)

        assert x.dtype == dtype
        return x, img

@persistence.persistent_class
class DepthPredictionBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        tmp_channels,                       # Number of intermediate channels.
        out_channels,                       # Number of output channels.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of input color channels.
        first_layer_idx,                    # Index of the first layer.
        architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet'.
        activation          = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        freeze_layers       = 0,            # Freeze-D: Number of layers to freeze.
    ):
        # assert in_channels in [0, tmp_channels, 2*tmp_channels]
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        self.num_layers = 0
        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable
        trainable_iter = trainable_gen()

        self.conv_compress = Conv2dLayer(in_channels, tmp_channels, kernel_size=1, activation=activation,
                                         trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        # Two Layer Simple Conv
        self.conv0 = Conv2dLayer(tmp_channels, tmp_channels, kernel_size=3, activation='linear',
            trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv1 = Conv2dLayer(tmp_channels, out_channels, kernel_size=3, activation='linear', up=1,
            trainable=next(trainable_iter), resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last)
        # Use upfird2d or Conv2d?
        if architecture == 'resnet':
            self.skip = Conv2dLayer(tmp_channels, out_channels, kernel_size=1, bias=False, up=1,
                trainable=next(trainable_iter), resample_filter=resample_filter, channels_last=self.channels_last)


    def forward(self, x, force_fp32=False):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
        x = x.to(dtype=dtype, memory_format=memory_format)

        # Compress
        x = self.conv_compress(x)

        # Main layers.
        if self.architecture == 'resnet':
            y = self.conv0(x, gain=np.sqrt(0.5))
            x.add_(y)
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv1(x, gain=np.sqrt(0.5))
            x.add_(y)
        else:
            x = self.conv0(x)
            x = self.conv1(x)

        assert x.dtype == dtype
        return x

@persistence.persistent_class
class SwitchDiscriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        depth_bit           =16,        # Depth Map Bit
        num_fp16_res        = 0,        # Use FP16 for the N highest resolutions.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        resample_filter     =[1,3,3,1], # For upfird2d.upsample2d
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
    ):
        super().__init__()
        # Original StyleGAN Discriminator (depth input layer added)
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        self.resample_filter = resample_filter
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlockDepth(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)

    # Depth Prediction Blocks
        self.up_resolutions = [2 ** i for i in range(4,self.img_resolution_log2+1)]
        for res in self.up_resolutions:
            in_channels = channels_dict[res] + channels_dict[res//2] if res>16 and res<self.img_resolution else channels_dict[res]
            # no img_resolution feature
            if res == img_resolution:
                in_channels = channels_dict[res//2]
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            block = DepthPredictionBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'd{res}', block)
            cur_layer_idx += block.num_layers

    # Final Depth Map Output
        self.conv_last = Conv2dLayer(channels_dict[img_resolution], depth_bit, kernel_size=3, activation='linear', up=1,
             conv_clamp=conv_clamp)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, img, c, mod, **block_kwargs): # mod(str) in ['depth', 'score']
        assert mod in ['depth','score']
        if mod == 'score':
            x = None
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                x, img = block(x, img, **block_kwargs)

            cmap = None
            if self.c_dim > 0:
                cmap = self.mapping(None, c)
            x = self.b4(x, img, cmap)
            return x

        else:
            x = None
            temp_feature = {}
            down_resolutions = [2 ** i for i in range(self.img_resolution_log2,4,-1)]
            for res in down_resolutions:
                block = getattr(self, f'b{res}')
                x, img = block(x, img, **block_kwargs) # Save Features
                temp_feature[res//2] = x

            for res in self.up_resolutions:
                if res == 16:
                    block = getattr(self, f'd{res}')
                    x = block(temp_feature[res], **block_kwargs)

                elif res == self.img_resolution:
                    x = F.interpolate(x,scale_factor=2)
                    block = getattr(self, f'd{res}')
                    x = block(x, **block_kwargs)

                else:
                    x = F.interpolate(x, scale_factor=2)
                    x = torch.cat((x,temp_feature[res]),dim=1) # cat along channel
                    block = getattr(self, f'd{res}')
                    x = block(x, **block_kwargs)

            # Final Depth Output

            x = self.conv_last(x)
            out = self.softmax(x)
            return out







