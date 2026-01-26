import torch
import torch.nn as nn
from models.cross_att import Basic_block
import torch.nn.functional as F
from models.linear import Linear
from models.linear import Linear_CNN
import models
from models import register
from einops import rearrange


class MultiAxisPooling(nn.Module):
    """Multi-axis pooling operation combining max and average pooling"""

    def __init__(self):
        super(MultiAxisPooling, self).__init__()

    def forward(self, x):
        # x shape: (batch, merged_channel_axis, dim2, dim3)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]  # Max pooling along first dimension
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # Average pooling along first dimension
        return torch.cat([max_pool, avg_pool], dim=1)  # Concatenate along channel dimension


class AxisAwareRefinement(nn.Module):
    """Axis-aware Visual Refinement Module (AVRM)"""

    def __init__(self, in_channels=768, reduction_ratio=16):
        super(AxisAwareRefinement, self).__init__()
        self.in_channels = in_channels

        # Shared convolution for all axes
        # Input channels: 2 (max + avg), output channels: 1
        self.conv3d = nn.Conv3d(2, 1, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm3d(1)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, channels, height, width, depth)
               Expected: (1, 768, 20, 20, 20)
        Returns:
            refined_x: Refined tensor of same shape
        """
        batch_size, c, h, w, d = x.shape

        # Create three directional views by binding channel with each spatial axis
        # View 1: bind channel with height -> (c*h, w, d)
        F_h = x.permute(0, 2, 1, 3, 4).contiguous()  # (1, h, c, w, d)
        F_h = F_h.view(batch_size, c * h, w, d)  # (1, c*h, w, d)

        # View 2: bind channel with width -> (c*w, h, d)
        F_w = x.permute(0, 3, 1, 2, 4).contiguous()  # (1, w, c, h, d)
        F_w = F_w.view(batch_size, c * w, h, d)  # (1, c*w, h, d)

        # View 3: bind channel with depth -> (c*d, h, w)
        F_d = x.permute(0, 4, 1, 2, 3).contiguous()  # (1, d, c, h, w)
        F_d = F_d.view(batch_size, c * d, h, w)  # (1, c*d, h, w)

        # Process each view with axis-aware refinement
        refined_views = []
        original_views = [F_h, F_w, F_d]

        for i, F_x in enumerate(original_views):
            # Multi-axis pooling along the merged channel-axis dimension
            # F_x shape: (1, merged_dim, dim2, dim3)
            pooled = MultiAxisPooling()(F_x)  # Output: (1, 2, dim2, dim3)

            # Add dummy spatial dimensions to make it 5D for 3D convolution
            # From (1, 2, dim2, dim3) to (1, 2, 1, dim2, dim3)
            pooled = pooled.unsqueeze(2)

            # Generate gating coefficients
            gate = self.conv3d(pooled)  # (1, 1, 1, dim2, dim3)
            gate = self.bn(gate)
            gate = torch.sigmoid(gate)

            # Remove the added dimension: (1, 1, 1, dim2, dim3) -> (1, 1, dim2, dim3)
            gate = gate.squeeze(2)

            # Apply gating to original feature view
            refined_view = F_x * gate  # Element-wise multiplication

            # Store refined view
            refined_views.append(refined_view)

        # Reshape refined views back to original volumetric format
        # Refined F_h: (1, c*h, w, d) -> (1, h, c, w, d) -> (1, c, h, w, d)
        F_h_refined = refined_views[0].view(batch_size, h, c, w, d).permute(0, 2, 1, 3, 4)

        # Refined F_w: (1, c*w, h, d) -> (1, w, c, h, d) -> (1, c, h, w, d)
        F_w_refined = refined_views[1].view(batch_size, w, c, h, d).permute(0, 2, 3, 1, 4)

        # Refined F_d: (1, c*d, h, w) -> (1, d, c, h, w) -> (1, c, h, w, d)
        F_d_refined = refined_views[2].view(batch_size, d, c, h, w).permute(0, 2, 3, 4, 1)

        # Aggregate the three enhanced views by averaging
        refined_x = (F_h_refined + F_w_refined + F_d_refined) / 3.0

        return refined_x
@register('rtfsyn')
class RTFSYN(nn.Module):

    def __init__(self, encoder_spec, no_imnet):
        super().__init__()
        self.encoder = models.make(encoder_spec)
        self.f_dim = self.encoder.out_dim
        self.fusion = Basic_block(dim=self.f_dim, num_heads=8)
        self.avrm = AxisAwareRefinement(in_channels=768)

        if no_imnet:
            self.imnet = None
        else:
            self.imnet = Linear(in_dim=self.f_dim+3,out_dim=1,hidden_list=[3072, 3072, 768, 256])

    def gen_feat(self, inp):
        feat = self.encoder(inp)
        return feat

    
    #tarin together
    def forward(self, src_lr, tgt_lr, coord_hr, prompt_src, prompt_tgt):
        N, K = coord_hr.shape[:2]
        feat_src_lr = self.avrm(self.gen_feat(src_lr)) # Extract features from the low-resolution input image

        feat_src_lr_tgt = self.fusion(feat_src_lr, prompt_tgt) #Fuse extracted features with the target image prompt
        vector_src_tgt = F.grid_sample(feat_src_lr_tgt, coord_hr.flip(-1).unsqueeze(1).unsqueeze(1), # Sample local feature vectors from the fused low-resolution feature map at the given high-resolution coordinates
                                       mode='bilinear',
                                       align_corners=False)[:, :, 0, 0, :].permute(0, 2, 1)
        vector_src_tgt_with_coord = torch.cat([vector_src_tgt, coord_hr], dim=-1) # Concatenate the sampled feature vectors with their corresponding normalized coordinates
        pre_src_tgt = self.imnet(vector_src_tgt_with_coord.view(N * K, -1)).view(N, K, -1) # Predict intensity values at queried coordinates using the implicit decoder

        feat_tgt_lr = self.avrm(self.gen_feat(tgt_lr))
        feat_tgt_lr_src = self.fusion(feat_tgt_lr, prompt_src)
        vector_tgt_src = F.grid_sample(feat_tgt_lr_src, coord_hr.flip(-1).unsqueeze(1).unsqueeze(1),
                                       mode='bilinear',
                                       align_corners=False)[:, :, 0, 0, :].permute(0, 2, 1)
        vector_tgt_src_with_coord = torch.cat([vector_tgt_src, coord_hr], dim=-1)
        pre_tgt_src = self.imnet(vector_tgt_src_with_coord.view(N * K, -1)).view(N, K, -1)
        return pre_src_tgt, pre_tgt_src, feat_src_lr, feat_tgt_lr
        

    # test forward
    def single_forward(self, src_lr, coord_hr, prompt_tgt):
        N, K = coord_hr.shape[:2]
        feat_src_lr = self.gen_feat(src_lr)
        feat_src_lr_tgt = self.fusion(feat_src_lr, prompt_tgt)
        vector_src_tgt = F.grid_sample(feat_src_lr_tgt, coord_hr.flip(-1).unsqueeze(1).unsqueeze(1),
                                       mode='bilinear',
                                       align_corners=False)[:, :, 0, 0, :].permute(0, 2, 1)
        vector_src_tgt_with_coord = torch.cat([vector_src_tgt, coord_hr], dim=-1)
        pre_src_tgt = self.imnet(vector_src_tgt_with_coord.view(N * K, -1)).view(N, K, -1)
        return pre_src_tgt


       
