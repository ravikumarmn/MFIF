import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class GenCleanBlock(nn.Module):
    """GenClean Block for denoising and preprocessing"""

    def __init__(self, in_channels, out_channels):
        super(GenCleanBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Residual connection if dimensions match
        if residual.size(1) == out.size(1):
            out += residual

        return self.relu(out)


class DeepFeatureExtractionBlock(nn.Module):
    """Deep Feature Extraction Block (DFEB) for enhanced feature learning"""

    def __init__(self, in_channels, out_channels, num_blocks=3):
        super(DeepFeatureExtractionBlock, self).__init__()

        # Multi-scale convolution branches
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, 3, padding=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, 3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, 3, padding=4, dilation=4),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
        )

        # Feature refinement
        self.refinement = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )

        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        # Multi-scale feature extraction
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        # Concatenate multi-scale features
        multi_scale = torch.cat([b1, b2, b3, b4], dim=1)

        # Feature refinement
        refined = self.refinement(multi_scale)

        # Residual connection
        residual = self.residual(x)

        return F.relu(refined + residual)


class DynamicChannelAdjustment(nn.Module):
    """Dynamic Channel Adjustment (DCA) for adaptive feature weighting"""

    def __init__(self, channels, reduction=16):
        super(DynamicChannelAdjustment, self).__init__()

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3), nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()

        # Channel attention
        channel_weights = self.gap(x).view(b, c)
        channel_weights = self.channel_attention(channel_weights).view(b, c, 1, 1)

        # Apply channel attention
        x_channel = x * channel_weights

        # Spatial attention
        avg_pool = torch.mean(x_channel, dim=1, keepdim=True)
        max_pool, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_weights = self.spatial_attention(spatial_input)

        # Apply spatial attention
        x_spatial = x_channel * spatial_weights

        return x_spatial


class SiameseEncoder(nn.Module):
    """Enhanced Siamese Encoder with DFEB blocks"""

    def __init__(self, in_channels=3):
        super(SiameseEncoder, self).__init__()

        # Initial GenClean Block
        self.gcb = GenCleanBlock(in_channels, 64)

        # Enhanced encoder layers with DFEB
        self.encoder1 = nn.Sequential(
            DeepFeatureExtractionBlock(64, 64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(2),
            DeepFeatureExtractionBlock(64, 128),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.encoder3 = nn.Sequential(
            nn.MaxPool2d(2),
            DeepFeatureExtractionBlock(128, 256),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.encoder4 = nn.Sequential(
            nn.MaxPool2d(2),
            DeepFeatureExtractionBlock(256, 512),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Apply GenClean Block
        x = self.gcb(x)

        # Encoder forward pass
        e1 = self.encoder1(x)  # 64 channels
        e2 = self.encoder2(e1)  # 128 channels
        e3 = self.encoder3(e2)  # 256 channels
        e4 = self.encoder4(e3)  # 512 channels

        return e1, e2, e3, e4


class AttentionFusion(nn.Module):
    """Enhanced attention-based fusion module with DCA"""

    def __init__(self, channels):
        super(AttentionFusion, self).__init__()

        # Enhanced attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

        # Dynamic Channel Adjustment
        self.dca = DynamicChannelAdjustment(channels)

    def forward(self, feat_a, feat_b):
        # Concatenate features
        concat_feat = torch.cat([feat_a, feat_b], dim=1)

        # Generate attention weights
        attention_weights = self.attention(concat_feat)

        # Apply attention to fuse features
        fused_feat = feat_a * attention_weights + feat_b * (1 - attention_weights)

        # Apply Dynamic Channel Adjustment
        fused_feat = self.dca(fused_feat)

        return fused_feat


class Discriminator(nn.Module):
    """GAN Discriminator for adversarial training"""

    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)


class VGGPerceptualLoss(nn.Module):
    """VGG-based perceptual loss for better visual quality"""

    def __init__(self, feature_layers=[3, 8, 15, 22]):
        super(VGGPerceptualLoss, self).__init__()

        # Load pre-trained VGG19
        vgg = models.vgg19(pretrained=True).features

        # Freeze VGG parameters
        for param in vgg.parameters():
            param.requires_grad = False

        self.feature_layers = feature_layers
        self.vgg_layers = nn.ModuleList()

        # Extract specific layers
        layer_idx = 0
        for i, layer in enumerate(vgg):
            self.vgg_layers.append(layer)
            if i in feature_layers:
                layer_idx += 1
            if layer_idx >= len(feature_layers):
                break

        # Normalization for VGG input
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def normalize_input(self, x):
        # Normalize input to [0, 1] range if needed
        if x.min() < 0:
            x = (x + 1) / 2  # Convert from [-1, 1] to [0, 1]

        # Apply VGG normalization
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        # Normalize inputs
        pred_norm = self.normalize_input(pred)
        target_norm = self.normalize_input(target)

        # Extract features
        pred_features = []
        target_features = []

        x_pred = pred_norm
        x_target = target_norm

        for i, layer in enumerate(self.vgg_layers):
            x_pred = layer(x_pred)
            x_target = layer(x_target)

            if i in self.feature_layers:
                pred_features.append(x_pred)
                target_features.append(x_target)

        # Calculate perceptual loss
        loss = 0
        for pred_feat, target_feat in zip(pred_features, target_features):
            loss += F.mse_loss(pred_feat, target_feat)

        return loss / len(pred_features)


class SiameseMFIFGAN(nn.Module):
    """Complete Siamese MFIF Network with GAN components for Phase 2"""

    def __init__(self, in_channels=3, out_channels=3):
        super(SiameseMFIFGAN, self).__init__()

        # Generator (enhanced MFIF network)
        self.generator = SiameseMFIF(in_channels, out_channels)

        # Discriminator
        self.discriminator = Discriminator(out_channels)

        # Perceptual loss
        self.perceptual_loss = VGGPerceptualLoss()

    def forward(self, source_a, source_b, mode="generator"):
        if mode == "generator":
            return self.generator(source_a, source_b)
        elif mode == "discriminator":
            fused_image = self.generator(source_a, source_b)
            return self.discriminator(fused_image)
        else:
            raise ValueError("Mode must be 'generator' or 'discriminator'")


class FusionDecoder(nn.Module):
    """Decoder for image reconstruction"""

    def __init__(self, out_channels=3):
        super(FusionDecoder, self).__init__()

        # Attention fusion modules
        self.fusion4 = AttentionFusion(512)
        self.fusion3 = AttentionFusion(256)
        self.fusion2 = AttentionFusion(128)
        self.fusion1 = AttentionFusion(64)

        # Decoder layers
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.decoder3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),  # 256 + 256 from skip connection
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.decoder2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),  # 128 + 128 from skip connection
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),  # 64 + 64 from skip connection
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, feat_a_list, feat_b_list):
        e1_a, e2_a, e3_a, e4_a = feat_a_list
        e1_b, e2_b, e3_b, e4_b = feat_b_list

        # Fuse features at each level
        f4 = self.fusion4(e4_a, e4_b)
        f3 = self.fusion3(e3_a, e3_b)
        f2 = self.fusion2(e2_a, e2_b)
        f1 = self.fusion1(e1_a, e1_b)

        # Decoder forward pass with skip connections
        d4 = self.decoder4(f4)
        d3 = self.decoder3(torch.cat([d4, f3], dim=1))
        d2 = self.decoder2(torch.cat([d3, f2], dim=1))
        output = self.decoder1(torch.cat([d2, f1], dim=1))

        return output


class SiameseMFIF(nn.Module):
    """Complete Siamese Multi-Focus Image Fusion Network"""

    def __init__(self, in_channels=3, out_channels=3):
        super(SiameseMFIF, self).__init__()

        # Shared Siamese encoder
        self.encoder = SiameseEncoder(in_channels)

        # Fusion decoder
        self.decoder = FusionDecoder(out_channels)

    def forward(self, source_a, source_b):
        # Extract features from both images using shared encoder
        feat_a_list = self.encoder(source_a)
        feat_b_list = self.encoder(source_b)

        # Fuse features using decoder
        fused_image = self.decoder(feat_a_list, feat_b_list)

        return fused_image
