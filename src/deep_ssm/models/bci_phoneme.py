

import torch 
import torch.nn as nn
from torch import Tensor
import math

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from deep_ssm.data.data_transforms import GaussianSmoothing
import torch.nn.functional as F
from deep_ssm.mixers.mamba_extra import MixerModel
from deep_ssm.mixers.s5 import S5Model
from deep_ssm.mixers.mamba2_simple import Mamba2Simple
from deep_ssm.models.audio_models import Sashimi
'''
Code adapted from Fracois Porcher: https://github.com/FrancoisPorcher/vit-pytorch
and from bit.py in transformers_with_dietcorp
'''

       
def pad_to_multiple(tensor, multiple, dim=1, value=0):
    """
    Pads `tensor` along `dim` so that its size is divisible by `multiple`.
    """
    size = tensor.size(dim)
    padding_needed = (multiple - size % multiple) % multiple
    if padding_needed == 0:
        return tensor
    pad_dims = [0] * (2 * tensor.dim())
    pad_dims[-2 * dim - 1] = padding_needed  # padding at the end
    return F.pad(tensor, pad_dims, value=value)



class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
def pair(t):
    if len(t) == 2:
        return t[0], t[1]
    elif isinstance(t, list):
        return t[0], t[1]
    elif isinstance(t, tuple):
        pass
    else:
        t = (t, t)
    return t

def get_sinusoidal_pos_emb(seq_len, dim, device=None):
    position = torch.arange(seq_len, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device) * -(math.log(10000.0) / dim))
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def create_temporal_mask(seq_len, look_ahead=0, device=None):
    i = torch.arange(seq_len, device=device).unsqueeze(1)
    j = torch.arange(seq_len, device=device).unsqueeze(0)
    mask = j <= i + look_ahead
    return mask.unsqueeze(0).unsqueeze(0)

class Attention(nn.Module):
    """
    Attenton module
    Args:
        dim: dimension of the input
        heads: number of heads
        dim_head: dimension of each head
        dropout: dropout rate is pre-attention dropout
        max_rel_dist: maximum relative distance for relative position bias
        use_relative_bias: whether to use relative position bias

    Returns:
        Tensor: (B, N, D)
    """
    def __init__(self, dim, heads, dim_head, dropout, max_rel_dist=200, use_relative_bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        # T5-style relative position bias
        self.max_rel_dist = max_rel_dist
        self.use_relative_bias = use_relative_bias
        if self.use_relative_bias:
            self.rel_pos_bias = nn.Embedding(2 * max_rel_dist - 1, 1)
      
    def forward(self, x, temporal_mask=None):
        """
        Process description mathematically:
        q = W_q * x
        k = W_k * x
        v = W_v * x
        dots = q * k.transpose(-1, -2)
        dots = dots + rel_pos_bias
        dots = dots.masked_fill(temporal_mask == 0, float('-inf'))
        attn = softmax(dots)
        Args:
            x: (B, N, D)
            temporal_mask: (B, N)
        Returns:
            Tensor: (B, N, D)
        """
        x = self.norm(x)
        # Project the input to q, k, v
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (b, h, n, n)

        # Add relative positional bias if enabled
        if self.use_relative_bias:
            seq_len = x.size(1)
            i = torch.arange(seq_len, device=x.device).unsqueeze(1)
            j = torch.arange(seq_len, device=x.device).unsqueeze(0)
            rel_pos = (i - j).clamp(-self.max_rel_dist + 1, self.max_rel_dist - 1) + self.max_rel_dist - 1
            rel_bias = self.rel_pos_bias(rel_pos).squeeze(-1).unsqueeze(0).unsqueeze(0) # shap seq_len x seq_len
            dots = dots + rel_bias

        if temporal_mask is not None:
            dots = dots.masked_fill(temporal_mask == 0, float('-inf'))
            
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim_ratio, 
                 dropout=0., use_relative_bias=True):
        
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        mlp_dim = mlp_dim_ratio * dim
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, heads=heads, dim_head=dim_head, 
                          dropout=dropout, use_relative_bias=use_relative_bias),
                FFN(dim=dim, hidden_dim=mlp_dim, dropout=dropout)
            ]))

    def forward(self, x, mask=None):
        for attn, ffn in self.layers:
            x = attn(x, temporal_mask=mask) + x
            x = ffn(x) + x
        return self.norm(x)
    
class BiT_Phoneme(nn.Module):
    
    def __init__(self, *, patch_size, dim, depth, heads, mlp_dim_ratio,
                 dim_head, dropout, input_dropout, look_ahead, gaussianSmoothWidth, 
                 nClasses, T5_style_pos, max_mask_pct, num_masks, mask_token_zeros,
                 num_masks_channels, max_mask_channels, dist_dict_path, linderman_lab):
   
        super().__init__()

        patch_height, patch_width = pair(patch_size)
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.dim = dim
        self.look_ahead = look_ahead  
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.T5_style_pos = T5_style_pos
        self.max_mask_pct = max_mask_pct
        self.num_masks = num_masks    
        self.patch_dim = patch_height * patch_width
        self.num_masks_channels = num_masks_channels
        self.max_channels_to_mask = max_mask_channels
        self.dist_dict_path = dist_dict_path
        self.linderman_lab = linderman_lab
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                    p1=patch_height, p2=patch_width),
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, dim),
            nn.LayerNorm(dim)
        )
        
        # Patch embedding split from encoder
        self.to_patch = self.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*self.to_patch_embedding[1:])

                
        self.gaussianSmoother = GaussianSmoothing(
            patch_width, 20, self.gaussianSmoothWidth, dim=1
        )
        
        if mask_token_zeros:
            self.mask_token = nn.Parameter(torch.zeros(self.patch_dim), requires_grad=False)
        else:
            self.mask_token = nn.Parameter(torch.randn(self.patch_dim))
                
        self.dropout = nn.Dropout(input_dropout)
  
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim_ratio, 
                                    dropout, use_relative_bias=self.T5_style_pos)
    
        self.projection = nn.Linear(dim, nClasses+1)
        
        if self.T5_style_pos == False:
            print("NOT USING T5 STYLE POS")
            self.register_buffer('pos_embedding', None, persistent=False)
        
        if self.linderman_lab:
            
            self.post_model_block = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Dropout(dropout),
                nn.Linear(dim, dim),
                nn.ReLU(),
            )
            
        else:
            
            self.post_model_block = nn.Identity()
        
    def forward(self, neuralInput, X_len, day_idx=None):
        """
        Args:
            neuralInput: Tensor of shape (B, T, F)
            X_len:Tensor of shape 
            dayIdx: tensor of shape (B)
        Returns:
            Tensor: (B, num_patches, dim)
        """
        
        neuralInput = pad_to_multiple(neuralInput, multiple=self.patch_height)
        
        #if self.training and self.max_channels_to_mask > 0: 
        #    neuralInput, _ = self.apply_channel_mask(neuralInput)
        
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
    

        neuralInput = neuralInput.unsqueeze(1)
        
        # add time masking
        if self.training and self.max_mask_pct > 0:

            x = self.to_patch(neuralInput)
            x, _ = self.apply_time_mask(x, X_len)    
            x = self.patch_to_emb(x)

        else:

            x = self.to_patch_embedding(neuralInput)

        # apply input level dropout. 
        x = self.dropout(x)
        
        b, seq_len, _ = x.shape
        
        # Add sin embeddings if T5 Style is False. 
        if self.T5_style_pos == False:
            pos_emb = get_sinusoidal_pos_emb(seq_len, self.dim, device=x.device)
            x = x + pos_emb.unsqueeze(0)
        
        # Create temporal mask
        temporal_mask = create_temporal_mask(seq_len, look_ahead=self.look_ahead, device=x.device)

        x = self.transformer(x, mask=temporal_mask)
        
        x = self.post_model_block(x)
        
        out = self.projection(x)
        
        return out
    
    def compute_length(self, X_len):
        
        # computing ceiling because I pad X to be divisible by path_height
        return torch.ceil(X_len / self.patch_height).to(dtype=torch.int32)
    
    
    def apply_original_augs(self, neuralInput, n_masks_nptl_augs, aug_values):
        
            device = neuralInput.device
        
            neuralInput = neuralInput.repeat_interleave(n_masks_nptl_augs, dim=0) 
            neuralInput += torch.randn(neuralInput.shape, 
                        device=device) * aug_values[0]
        
            neuralInput += (
                torch.randn([neuralInput.shape[0], 1, neuralInput.shape[2]], 
                device=device)
                * aug_values[1]
            )
            
            return neuralInput
        
        
    def apply_time_mask(self, X, X_len, constant_mask=False, mask_range=[]):
        
        """
        Fully vectorized SpecAugment-style time masking (no loops at all).
        
        Args:
            X: (B, P, D) input tensor
            X_len: (B,) valid lengths in timepoints
            constant_mask_lengths: if True, make the mask lengths the same across all batches

        Returns:
            X_masked: (B, P, D) with masked patches
            mask: (B, P) boolean mask of where values were masked
            masked_indices: list of 1D LongTensors, each with indices of masked patches per batch
            unmasked_indices: list of 1D LongTensors, each with indices of unmasked patches per batch
        """
        B, P, D = X.shape
        device = X.device

        if constant_mask:
            # get valid len of smallest trial in batch and repeat for all batches. 
            valid_lens = torch.min((X_len // self.patch_height).to(device)).repeat(B)
        else:
            valid_lens = (X_len // self.patch_height).to(device)
            
        max_mask_lens = (self.max_mask_pct * valid_lens).long()  # (B,)

        # Repeat B num_masks times to simulate multiple masks per sample
        B_rep = B * self.num_masks

        # Expand inputs for vectorized masking
        # repeat_interleave works like tile, so values corresponding to the same batch are next to each other
        valid_lens_rep = valid_lens.repeat_interleave(self.num_masks)            # (B * num_masks,)
        max_mask_lens_rep = max_mask_lens.repeat_interleave(self.num_masks)      # (B * num_masks,)

        if constant_mask:
            # select the same t for every batch. 
            t = (torch.rand(self.num_masks, device=device).repeat(B) * (max_mask_lens_rep + 1).float()).floor().long().clamp(min=1)  # (B * num_masks,)
        else:
            t = (torch.rand(B_rep, device=device) * (max_mask_lens_rep + 1).float()).floor().long()  # (B * num_masks,)
            
        max_start = (valid_lens_rep - t + 1).clamp(min=1)
        
        if constant_mask:
            t0 = (torch.rand(self.num_masks, device=device).repeat(B) * max_start.float()).floor().long()               # (B * num_masks,)
        else:
            t0 = (torch.rand(B_rep, device=device) * max_start.float()).floor().long()               # (B * num_masks,)

        # Build the global mask (B, P)
        arange = torch.arange(P, device=device).unsqueeze(0)       # (1, P)
        t0_exp = t0.unsqueeze(1)                                   # (B_rep, 1)
        t1_exp = (t0 + t).unsqueeze(1)                             # (B_rep, 1)
        mask_chunks = (arange >= t0_exp) & (arange < t1_exp)       # (B_rep, P)
        
        # Get index of sample in batch for each mask chunk
        batch_idx = torch.arange(B, device=device).repeat_interleave(self.num_masks)  # (B * num_masks,)

        # Now scatter all the masks into the full mask (B, P)
        patch_idx = mask_chunks.nonzero(as_tuple=False)  # (N_masked, 2)
        b_indices = batch_idx[patch_idx[:, 0]]           # (N_masked,)
        p_indices = patch_idx[:, 1]                      # (N_masked,)

        mask = torch.zeros(B, P, dtype=torch.bool, device=device)
        mask[b_indices, p_indices] = True
        
        # mask: (B, P) boolean, True for masked
        #B, P = mask.shape

        # Number of masked patches per batch (assumed same for all batches)
        if constant_mask:
            N = mask.sum(dim=1)[0].item()
            U = P - N  # Number of unmasked per batch
                            
            masked_indices = mask.nonzero(as_tuple=False)  # (B * N, 2) — rows: [batch_idx, patch_idx]
            masked_indices = masked_indices[:, 1].reshape(B, N)
            masked_indices = torch.sort(masked_indices, dim=-1).values  # sort within batch
        
            unmasked = ~mask  # invert the mask
            unmasked_indices = unmasked.nonzero(as_tuple=False)[:, 1].reshape(B, U)
            unmasked_indices = torch.sort(unmasked_indices, dim=-1).values
        
            return masked_indices, unmasked_indices
        
        # Apply the mask
        X_masked = X.clone()
        X_masked[mask] = self.mask_token

        return X_masked, mask
    
class BiT_Mamba_Phoneme(BiT_Phoneme):
    """BiT model that uses Mamba instead of Transformer layers"""
    
    def __init__(self, *, patch_size, dim, d_model, d_state, d_conv, expand_factor, 
                 layer_dim, bidirectional, bidirectional_strategy, mamba_bi_new,
                 dropout, input_dropout, look_ahead, gaussianSmoothWidth, 
                 nClasses, T5_style_pos, max_mask_pct, num_masks, mask_token_zeros,
                 num_masks_channels, max_mask_channels, dist_dict_path, linderman_lab,
                 fused_add_norm=False, rms_norm=False, initialize_mixer=False,
                 # Additional transformer parameters needed for parent class
                 depth=None, heads=None, mlp_dim_ratio=None, dim_head=None):
        
        # Set default transformer parameters (not used but required by parent)
        if depth is None:
            depth = layer_dim
        if heads is None:
            heads = 8
        if mlp_dim_ratio is None:
            mlp_dim_ratio = 4
        if dim_head is None:
            dim_head = 64
            
        # Initialize parent class first
        super().__init__(
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim_ratio=mlp_dim_ratio,
            dim_head=dim_head,
            dropout=dropout,
            input_dropout=input_dropout,
            look_ahead=look_ahead,
            gaussianSmoothWidth=gaussianSmoothWidth,
            nClasses=nClasses,
            T5_style_pos=T5_style_pos,
            max_mask_pct=max_mask_pct,
            num_masks=num_masks,
            mask_token_zeros=mask_token_zeros,
            num_masks_channels=num_masks_channels,
            max_mask_channels=max_mask_channels,
            dist_dict_path=dist_dict_path,
            linderman_lab=linderman_lab,
        )
        
        # Store mamba-specific parameters
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand_factor = expand_factor
        self.layer_dim = layer_dim
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        self.mamba_bi_new = mamba_bi_new
        
        # Replace transformer with mamba backbone
        del self.transformer  # Remove the transformer from parent class
        
        # Project from patch embedding dim to mamba d_model if different
        if dim != d_model:
            self.dim_projection = nn.Linear(dim, d_model)
        else:
            self.dim_projection = nn.Identity()
            
        # Mamba backbone
        self.mamba_backbone = MixerModel(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
            n_layer=layer_dim,
            rms_norm=rms_norm,
            fused_add_norm=fused_add_norm,
            bidirectional=bidirectional,
            mamba_bi_new=mamba_bi_new,
            initialize_mixer=initialize_mixer,
            bidirectional_strategy=bidirectional_strategy,
            #residual_in_fp32=residual_in_fp32,
        )
        
        # Output projection
        if bidirectional and bidirectional_strategy == "concatenate":
            d_output = d_model * 2
        else:
            d_output = d_model
            
        if d_output != dim:
            self.output_projection = nn.Linear(d_output, dim)
        else:
            self.output_projection = nn.Identity()
    
    def forward(self, neuralInput, X_len, day_idx=None):
        """
        Args:
            neuralInput: Tensor of shape (B, T, F)
            X_len: Tensor of shape (B,)
            day_idx: tensor of shape (B)
        Returns:
            Tensor: (B, num_patches, nClasses+1)
        """
        
        neuralInput = pad_to_multiple(neuralInput, multiple=self.patch_height)
        
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        
        neuralInput = neuralInput.unsqueeze(1)
        
        # add time masking
        if self.training and self.max_mask_pct > 0:
            x = self.to_patch(neuralInput)
            x, _ = self.apply_time_mask(x, X_len)    
            x = self.patch_to_emb(x)
        else:
            x = self.to_patch_embedding(neuralInput)

        # apply input level dropout. 
        x = self.dropout(x)
        
        b, seq_len, _ = x.shape
        
        # Add sin embeddings if T5 Style is False. 
        if self.T5_style_pos == False:
            pos_emb = get_sinusoidal_pos_emb(seq_len, self.dim, device=x.device)
            x = x + pos_emb.unsqueeze(0)
        
        # Project to mamba dimension
        x = self.dim_projection(x)  # (B, seq_len, d_model)
        
        # Pass through mamba backbone (no temporal mask needed for mamba)
        x = self.mamba_backbone(x)  # (B, seq_len, d_model or d_model*2)
        
        # Project back to dim
        x = self.output_projection(x)  # (B, seq_len, dim)
        
        # Apply post-model block
        x = self.post_model_block(x)
        
        # Final projection to classes
        out = self.projection(x)  # (B, seq_len, nClasses+1)
        
        return out

class BiT_S5_Phoneme(BiT_Phoneme):
    """BiT model that uses S5 instead of Transformer layers"""
    
    def __init__(self, *, patch_size, dim, d_model, d_state, n_layers, 
                 dropout, input_dropout, look_ahead, gaussianSmoothWidth, 
                 nClasses, T5_style_pos, max_mask_pct, num_masks, mask_token_zeros,
                 num_masks_channels, max_mask_channels, dist_dict_path, linderman_lab,
                 # S5-specific parameters
                 ssm_size=None, blocks=1, bidirectional=False, conj_sym=False,
                 prenorm=False, batchnorm=False, bn_momentum=0.95, bandlimit=None,
                 step_rescale=1.0, clip_eigs=False, activation="gelu", add_residual=True,
                 norm_epsilon=1e-5, rms_norm=False, C_init="lecun_normal",
                 # Additional transformer parameters needed for parent class
                 depth=None, heads=None, mlp_dim_ratio=None, dim_head=None):
        
        # Set default transformer parameters (not used but required by parent)
        if depth is None:
            depth = n_layers
        if heads is None:
            heads = 8
        if mlp_dim_ratio is None:
            mlp_dim_ratio = 4
        if dim_head is None:
            dim_head = 64
        if ssm_size is None:
            ssm_size = d_model
            
        # Initialize parent class first
        super().__init__(
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim_ratio=mlp_dim_ratio,
            dim_head=dim_head,
            dropout=dropout,
            input_dropout=input_dropout,
            look_ahead=look_ahead,
            gaussianSmoothWidth=gaussianSmoothWidth,
            nClasses=nClasses,
            T5_style_pos=T5_style_pos,
            max_mask_pct=max_mask_pct,
            num_masks=num_masks,
            mask_token_zeros=mask_token_zeros,
            num_masks_channels=num_masks_channels,
            max_mask_channels=max_mask_channels,
            dist_dict_path=dist_dict_path,
            linderman_lab=linderman_lab,
        )
        
        # Store S5-specific parameters
        self.d_model = d_model
        self.d_state = d_state
        self.n_layers = n_layers
        self.ssm_size = ssm_size
        self.blocks = blocks
        self.bidirectional = bidirectional
        self.conj_sym = conj_sym
        self.prenorm = prenorm
        self.batchnorm = batchnorm
        self.bn_momentum = bn_momentum
        self.bandlimit = bandlimit
        self.step_rescale = step_rescale
        self.clip_eigs = clip_eigs
        self.activation = activation
        self.add_residual = add_residual
        self.norm_epsilon = norm_epsilon
        self.rms_norm = rms_norm
        self.C_init = C_init
        
        # Replace transformer with S5 backbone
        del self.transformer  # Remove the transformer from parent class
        
        # Project from patch embedding dim to S5 d_model if different
        if dim != d_model:
            self.dim_projection = nn.Linear(dim, d_model)
        else:
            self.dim_projection = nn.Identity()
            
        # S5 backbone
        self.s5_backbone = S5Model(
            d_model=d_model,
            ssm_size=ssm_size,
            n_layers=n_layers,
            blocks=blocks,
            bidirectional=bidirectional,
            conj_sym=conj_sym,
            clip_eigs=clip_eigs,
            step_rescale=step_rescale,
            dropout=dropout,
            activation=activation,
            prenorm=prenorm,
            batchnorm=batchnorm,
            bn_momentum=bn_momentum,
            bandlimit=bandlimit,
            add_residual=add_residual,
            C_init=C_init,
        )
        
        # Final normalization
        if rms_norm:
            try:
                self.norm_f = nn.RMSNorm(d_model, eps=norm_epsilon)
            except AttributeError:
                # Fallback to LayerNorm if RMSNorm not available
                self.norm_f = nn.LayerNorm(d_model, eps=norm_epsilon)
        else:
            self.norm_f = nn.LayerNorm(d_model, eps=norm_epsilon)
        
        # Output projection
        if d_model != dim:
            self.output_projection = nn.Linear(d_model, dim)
        else:
            self.output_projection = nn.Identity()
    
    def forward(self, neuralInput, X_len, day_idx=None):
        """
        Args:
            neuralInput: Tensor of shape (B, T, F)
            X_len: Tensor of shape (B,)
            day_idx: tensor of shape (B)
        Returns:
            Tensor: (B, num_patches, nClasses+1)
        """
        
        neuralInput = pad_to_multiple(neuralInput, multiple=self.patch_height)
        
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        
        neuralInput = neuralInput.unsqueeze(1)
        
        # add time masking
        if self.training and self.max_mask_pct > 0:
            x = self.to_patch(neuralInput)
            x, _ = self.apply_time_mask(x, X_len)    
            x = self.patch_to_emb(x)
        else:
            x = self.to_patch_embedding(neuralInput)

        # apply input level dropout. 
        x = self.dropout(x)
        
        b, seq_len, _ = x.shape
        
        # Add sin embeddings if T5 Style is False. 
        if self.T5_style_pos == False:
            pos_emb = get_sinusoidal_pos_emb(seq_len, self.dim, device=x.device)
            x = x + pos_emb.unsqueeze(0)
        
        # Project to S5 dimension
        x = self.dim_projection(x)  # (B, seq_len, d_model)
        
        # Pass through S5 backbone (no temporal mask needed for S5)
        x, _ = self.s5_backbone(x)  # (B, seq_len, d_model)
        
        # Apply final normalization
        x = self.norm_f(x)
        
        # Project back to dim
        x = self.output_projection(x)  # (B, seq_len, dim)
        
        # Apply post-model block
        x = self.post_model_block(x)
        
        # Final projection to classes
        out = self.projection(x)  # (B, seq_len, nClasses+1)
        
        return out


class BiT_Mamba2_Phoneme(BiT_Phoneme):
    """BiT model that uses Mamba2 instead of Transformer layers"""
    
    def __init__(self, *, patch_size, dim, d_model, d_state, d_conv, expand_factor, 
                 layer_dim, dropout, input_dropout, look_ahead, gaussianSmoothWidth, 
                 nClasses, T5_style_pos, max_mask_pct, num_masks, mask_token_zeros,
                 num_masks_channels, max_mask_channels, dist_dict_path, linderman_lab,
                 # Additional transformer parameters needed for parent class
                 depth=None, heads=None, mlp_dim_ratio=None, dim_head=None):
        
        # Set default transformer parameters (not used but required by parent)
        if depth is None:
            depth = layer_dim
        if heads is None:
            heads = 8
        if mlp_dim_ratio is None:
            mlp_dim_ratio = 4
        if dim_head is None:
            dim_head = 64
            
        # Initialize parent class first
        super().__init__(
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim_ratio=mlp_dim_ratio,
            dim_head=dim_head,
            dropout=dropout,
            input_dropout=input_dropout,
            look_ahead=look_ahead,
            gaussianSmoothWidth=gaussianSmoothWidth,
            nClasses=nClasses,
            T5_style_pos=T5_style_pos,
            max_mask_pct=max_mask_pct,
            num_masks=num_masks,
            mask_token_zeros=mask_token_zeros,
            num_masks_channels=num_masks_channels,
            max_mask_channels=max_mask_channels,
            dist_dict_path=dist_dict_path,
            linderman_lab=linderman_lab,
        )
        
        # Store mamba2-specific parameters
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand_factor = expand_factor
        self.layer_dim = layer_dim
        
        # Replace transformer with mamba2 backbone
        del self.transformer  # Remove the transformer from parent class
        
        # Project from patch embedding dim to mamba2 d_model if different
        if dim != d_model:
            self.dim_projection = nn.Linear(dim, d_model)
        else:
            self.dim_projection = nn.Identity()
            
        # Mamba2 backbone
        from deep_ssm.mixers.mamba_extra import MixerModel
        self.mamba2_backbone = MixerModel(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
            n_layer=layer_dim,
            mamba_bi_new=True,
            mamba2=True,
        )
        
        # Output projection
        if d_model != dim:
            self.output_projection = nn.Linear(d_model, dim)
        else:
            self.output_projection = nn.Identity()
    
    def forward(self, neuralInput, X_len, day_idx=None):
        """
        Args:
            neuralInput: Tensor of shape (B, T, F)
            X_len: Tensor of shape (B,)
            day_idx: tensor of shape (B)
        Returns:
            Tensor: (B, num_patches, nClasses+1)
        """
        
        neuralInput = pad_to_multiple(neuralInput, multiple=self.patch_height)
        
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        
        neuralInput = neuralInput.unsqueeze(1)
        
        # add time masking
        if self.training and self.max_mask_pct > 0:
            x = self.to_patch(neuralInput)
            x, _ = self.apply_time_mask(x, X_len)    
            x = self.patch_to_emb(x)
        else:
            x = self.to_patch_embedding(neuralInput)

        # apply input level dropout. 
        x = self.dropout(x)
        
        b, seq_len, _ = x.shape
        
        # Add sin embeddings if T5 Style is False. 
        if self.T5_style_pos == False:
            pos_emb = get_sinusoidal_pos_emb(seq_len, self.dim, device=x.device)
            x = x + pos_emb.unsqueeze(0)
        
        # Project to mamba2 dimension
        x = self.dim_projection(x)  # (B, seq_len, d_model)
        
        # Pass through mamba2 backbone (no temporal mask needed for mamba2)
        x = self.mamba2_backbone(x)  # (B, seq_len, d_model)
        
        # Project back to dim
        x = self.output_projection(x)  # (B, seq_len, dim)
        
        # Apply post-model block
        x = self.post_model_block(x)
        
        # Final projection to classes
        out = self.projection(x)  # (B, seq_len, nClasses+1)
        
        return out



class BiT_Sashimi_Phoneme(BiT_Phoneme):
    """BiT model that uses Sashimi instead of Transformer layers"""
    
    def __init__(self, *, patch_size, dim, d_model, d_state, d_conv, expand_factor, 
                 layer_dim, dropout, input_dropout, look_ahead, gaussianSmoothWidth, 
                 nClasses, T5_style_pos, max_mask_pct, num_masks, mask_token_zeros,
                 num_masks_channels, max_mask_channels, dist_dict_path, linderman_lab,
                 # Additional transformer parameters needed for parent class
                 depth=None, heads=None, mlp_dim_ratio=None, dim_head=None,
                 ssm_type="s4", bidirectional=False, pool=[4, 4], ff=2, unet=False, transposed=True):
        
        # Set default transformer parameters (not used but required by parent)
        if depth is None:
            depth = layer_dim
        if heads is None:
            heads = 8
        if mlp_dim_ratio is None:
            mlp_dim_ratio = 4
        if dim_head is None:
            dim_head = 64

        # Initialize parent class first
        super().__init__(
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim_ratio=mlp_dim_ratio,
            dim_head=dim_head,
            dropout=dropout,
            input_dropout=input_dropout,
            look_ahead=look_ahead,
            gaussianSmoothWidth=gaussianSmoothWidth,
            nClasses=nClasses,
            T5_style_pos=T5_style_pos,
            max_mask_pct=max_mask_pct,
            num_masks=num_masks,
            mask_token_zeros=mask_token_zeros,
            num_masks_channels=num_masks_channels,
            max_mask_channels=max_mask_channels,
            dist_dict_path=dist_dict_path,
            linderman_lab=linderman_lab,
        )
        
        # Store sashimi-specific parameters
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand_factor = expand_factor
        self.layer_dim = layer_dim
        
        # Replace transformer with sashimi backbone
        del self.transformer  # Remove the transformer from parent class
        
        # Project from patch embedding dim to sashimi d_model if different
        if dim != d_model:
            self.dim_projection = nn.Linear(dim, d_model)
        else:
            self.dim_projection = nn.Identity()
            
        # Sashimi backbone
        self.sashimi_backbone = Sashimi(
            d_model=d_model,
            n_layers=layer_dim,
            pool=pool,
            expand=expand_factor,
            ff=ff,
            bidirectional=bidirectional,
            unet=unet,
            dropout=dropout,
            ssm_type=ssm_type,
            transposed=transposed,
        )
        
        # Output projection
        if d_model != dim:
            self.output_projection = nn.Linear(d_model, dim)
        else:
            self.output_projection = nn.Identity()
    
    def forward(self, neuralInput, X_len, day_idx=None):
        """
        Args:
            neuralInput: Tensor of shape (B, T, F)
            X_len: Tensor of shape (B,)
            day_idx: tensor of shape (B)
        Returns:
            Tensor: (B, num_patches, nClasses+1)
        """
        
        neuralInput = pad_to_multiple(neuralInput, multiple=self.patch_height)
        
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        
        neuralInput = neuralInput.unsqueeze(1)
        
        # add time masking
        if self.training and self.max_mask_pct > 0:
            x = self.to_patch(neuralInput)
            x, _ = self.apply_time_mask(x, X_len)    
            x = self.patch_to_emb(x)
        else:
            x = self.to_patch_embedding(neuralInput)

        # apply input level dropout. 
        x = self.dropout(x)
        
        b, seq_len, _ = x.shape
        
        # Add sin embeddings if T5 Style is False. 
        if self.T5_style_pos == False:
            pos_emb = get_sinusoidal_pos_emb(seq_len, self.dim, device=x.device)
            x = x + pos_emb.unsqueeze(0)
        
        # Project to sashimi dimension
        x = self.dim_projection(x)  # (B, seq_len, d_model)
        
        # Pass through sashimi backbone (no temporal mask needed for sashimi)
        x, _ = self.sashimi_backbone(x)  # (B, seq_len, d_model)
        
        # Project back to dim
        x = self.output_projection(x)  # (B, seq_len, dim)
        
        # Apply post-model block
        x = self.post_model_block(x)
        
        # Final projection to classes
        out = self.projection(x)  # (B, seq_len, nClasses+1)
        
        return out