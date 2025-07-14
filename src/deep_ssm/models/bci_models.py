from deep_ssm.data.data_transforms import GaussianSmoothing
import torch
import torch.nn as nn
from deep_ssm.mixers.mamba_extra import MixerModel
from deep_ssm.models.audio_models import Sashimi
import torch.nn.functional as F
from deep_ssm.mixers.s5 import S5Model
from typing import Optional, List
import sys
import os 
from deep_ssm.models.bci_phoneme import BiT_Phoneme, BiT_Mamba_Phoneme, BiT_S5_Phoneme, BiT_Mamba2_Phoneme

class OutputDecoder(nn.Module):
    def __init__(self, d_output, n_classes, init_output_decoder_layer=False):
        super(OutputDecoder, self).__init__()
        self.fc_decoder_out = nn.Linear(d_output, n_classes)  
        if init_output_decoder_layer:
            nn.init.xavier_uniform_(self.fc_decoder_out.weight)
            nn.init.zeros_(self.fc_decoder_out.bias)
        
    def forward(self, x):
        seq_out = self.fc_decoder_out(x)
        return seq_out


class BaseDecoder(nn.Module):
    def __init__(
        self,
        neural_dim,
        nDays=24,
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        gaussianSmoothSize = 20,
        unfolding=True,
        input_nonlinearity="softsign",
        add_embedding_layer=False,
        init_embedding_layer=False,
        embedding_layer_dim=None,

    ):
        """
        The BaseDecoder class is designed to process sequential
        It applies:
        Gaussian smoothing along features
          (neural_dim, gaussianSmoothWidth, gaussianSmoothSize)
        unfolds (extracts extracts sliding windows) along the sequence dimension:
          (kernelLen, strideLen)
        applies a day specific linear transformation and bias to the input
          (nDays)
        """
        super(BaseDecoder, self).__init__()

        self.neural_dim = neural_dim
        self.nDays = nDays
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.gaussianSmoothSize = gaussianSmoothSize
        self.unfolding = unfolding
        self.input_nonlinearity = input_nonlinearity
        self.add_embedding_layer = add_embedding_layer

        # Define the input layer nonlinearity (Softsign activation)
        if self.input_nonlinearity == "softsign":
            self.inputLayerNonlinearity = torch.nn.Softsign()

        # Define an unfold operation, which extracts sliding local blocks from a batched input tensor
        # This operation helps in simulating a convolution-like behavior with kernel and stride.
        if self.unfolding:
          self.unfolder = torch.nn.Unfold(
              (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
          )

        # If gaussian smoothing is applied, define a gaussian smoother using the specified width
        # the smoother is applied along the feature dimension
        if self.gaussianSmoothWidth > 0:
            self.gaussianSmoother = GaussianSmoothing(
                self.neural_dim, self.gaussianSmoothSize, self.gaussianSmoothWidth, dim=1
            )

        # Define day-specific weight matrices (learnable parameters) for transforming the input
        # There is one weight matrix per day, with dimensions neural_dim x neural_dim
        self.dayWeights = nn.Parameter(torch.randn(self.nDays, self.neural_dim, self.neural_dim))

        # Define day-specific biases (learnable parameters), one per day, with dimensions 1 x neural_dim
        self.dayBias = nn.Parameter(torch.zeros(self.nDays, 1, self.neural_dim))

        # Initialize dayWeights with identity matrices for each day (ensuring no transformation at the start)
        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(self.neural_dim)

        # Input layers
        #for x in range(nDays):
        #    setattr(self, "inpLayer" + str(x), nn.Linear(self.neural_dim, self.neural_dim))

        #for x in range(nDays):
        #    thisLayer = getattr(self, "inpLayer" + str(x))
        #    thisLayer.weight = nn.Parameter(thisLayer.weight + torch.eye(self.neural_dim))

        if self.add_embedding_layer:
            assert embedding_layer_dim is not None, "embedding_layer_dim must be provided if add_embedding_layer is True"
            if self.unfolding:
                in_dims = self.neural_dim * self.kernelLen
            else:
                in_dims = self.neural_dim
            self.embedding_layer = nn.Linear(in_dims, embedding_layer_dim)
            if init_embedding_layer:
                nn.init.xavier_uniform_(self.embedding_layer.weight)
                nn.init.zeros_(self.embedding_layer.bias)

    def forward_preprocessing(self, neuralInput, dayIdx):
        """

        Args:
          neuralInput: (batch_size x seq_len x num_features)
          dayIdx: (batch_size, )

        Returns:
          stridedInputs: (batch_size x new_seq_len x new_num_features)
        """
        # Smooth along the feature dimension
        if self.gaussianSmoothWidth > 0:
            neuralInput = torch.permute(neuralInput, (0, 2, 1))  # (BS, N, L)
            neuralInput = self.gaussianSmoother(neuralInput)
            neuralInput = torch.permute(neuralInput, (0, 2, 1))  # (BS, L, N)

        # Select the weight matrix for the current day based on dayIdx
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)  # (BS, N, N)

        # Apply a linear transformation to the neural input using the selected day weight matrix
        # This performs a batch-wise matrix multiplication followed by adding the corresponding day bias
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)  # (BS, L, N)

        # Map values between [-1, 1]
        if self.input_nonlinearity == "softsign":
            transformedNeural = self.inputLayerNonlinearity(transformedNeural)  # (BS, L, N)

        # Apply the unfold operation extracts sliding windows along seq dimension
        # the feature dimension is expanded by a factor of kernelLen
        # It essentially extracts overlapping blocks of size kernelLen with stride strideLen.
        if self.unfolding:
          stridedInputs = torch.permute(
              self.unfolder(
                  torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)  # (BS, N, L, 1)
              ),
              (0, 2, 1),
          )  # (BS, new L, new N)
        else:
          stridedInputs = transformedNeural

        #assert stridedInputs.shape == (neuralInput.shape[0], (neuralInput.shape[1] - self.kernelLen) // self.strideLen + 1, self.neural_dim*self.kernelLen)
        if self.add_embedding_layer:
            stridedInputs = self.embedding_layer(stridedInputs)
        return stridedInputs


class GRUDecoder(BaseDecoder):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        nDays=24,
        dropout=0,
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        unfolding=True,
        bidirectional=False,
        input_nonlinearity="softsign",
    ):
        super(GRUDecoder, self).__init__(
            neural_dim=neural_dim,
            nDays=nDays,
            strideLen=strideLen,
            kernelLen=kernelLen,
            gaussianSmoothWidth=gaussianSmoothWidth,
            unfolding=unfolding,
            input_nonlinearity=input_nonlinearity,
        )

        self.layer_dim = layer_dim
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        if unfolding:
          input_dims = self.neural_dim * kernelLen
        else:
          input_dims = self.neural_dim

        self.gru_decoder = nn.GRU(
            input_dims,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        for name, param in self.gru_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        self.fc_decoder_out = nn.Linear(
            self.hidden_dim * (2 if self.bidirectional else 1), n_classes + 1
        )  # +1 for CTC blank

    def forward(self, neuralInput, dayIdx):
        # Preprocess the input (e.g., Gaussian smoothing and unfolding)
        stridedInputs = self.forward_preprocessing(neuralInput, dayIdx)

        # Initialize hidden state
        h0 = torch.zeros(
            self.layer_dim * (2 if self.bidirectional else 1),
            stridedInputs.size(0),
            self.hidden_dim,
            device=neuralInput.device,
        ).requires_grad_()

        # Apply GRU Layer
        hid, _ = self.gru_decoder(stridedInputs, h0.detach())

        # Apply Decoder
        seq_out = self.fc_decoder_out(hid)
        return seq_out


class MambaDecoder(BaseDecoder):
    def __init__(
        self,
        neural_dim,
        n_classes,
        # model parameters
        d_model,
        d_state,
        d_conv,
        expand_factor,
        layer_dim,
        bidirectional_input=False,
        bidirectional=False,
        mamba_bi_new=True,
        fused_add_norm=False,
        rms_norm=False,
        initialize_mixer=False,
        bidirectional_strategy=None,
        dropout=0.0,
        # preprocessing parameters
        unfolding=True,
        nDays=24,
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        input_nonlinearity="softsign",
        # additional preprocessing parameters
        normalize_batch=False,
        init_embedding_layer=False,
        include_relu=False,
        # additional decoding layer:
        fcc_layers=False,
        activation="relu",
        dff=None,
        add_embedding_layer=True,
        init_output_decoder_layer=False,
    ):
        
        super(MambaDecoder, self).__init__(
            neural_dim=neural_dim,
            nDays=nDays,
            strideLen=strideLen,
            kernelLen=kernelLen,
            gaussianSmoothWidth=gaussianSmoothWidth,
            unfolding=unfolding,
            input_nonlinearity=input_nonlinearity,
            add_embedding_layer=add_embedding_layer,
            init_embedding_layer=init_embedding_layer,
            embedding_layer_dim=d_model,
        )
        self.layer_dim = layer_dim
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand_factor = expand_factor
        self.normalize_batch = normalize_batch
        self.include_relu = include_relu
        self.fcc_layers = fcc_layers

        if bidirectional_input:
            raise NotImplementedError("Bidirectional input not supported for MambaDecoder")

        # input dimension to model dimension
        self.dropout = nn.Dropout(p=dropout)

        # Block of model layers
        self.backbone = MixerModel(
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
        )

        # from model dimension to n_classes
        if bidirectional and bidirectional_strategy == "concatenate":
            d_output = d_model*2
        else:
            d_output = d_model

        self.fc_decoder_out = OutputDecoder(d_output, n_classes + 1, init_output_decoder_layer)  # +1 for CTC blank

        if self.fcc_layers:
            d_ff = dff or 4*d_output
            self.activation = F.relu if activation == 'relu' else F.gelu
            self.conv1 = nn.Conv1d(in_channels=d_output, out_channels=d_ff, kernel_size=1)
            self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_output, kernel_size=1)
            self.norm2 = nn.LayerNorm(d_output)


    def forward(self, neuralInput, dayIdx):
        """
        Forward pass of the Decoder
        Args:
            neuralInput: (batch_size x seq_len x num_features)
            dayIdx: (batch_size, )

        Returns:

        """
        if self.normalize_batch:
            dim_ = 1
            means = neuralInput.mean(dim_, keepdim=True).detach() # B x 1 x D
            neuralInput = neuralInput - means
            stdev = torch.sqrt(torch.var(neuralInput, dim=dim_, keepdim=True, unbiased=False) + 1e-5)  # B x 1 x D
            neuralInput /= stdev

        # Preprocess batch
        hidden_states = self.forward_preprocessing(neuralInput, dayIdx)

        # include relu
        if self.include_relu:
            hidden_states = torch.relu(hidden_states)
    
        # Pass through the mixer
        hidden_states = self.backbone(hidden_states)

        # Pass through FCC layers:
        if self.fcc_layers:
            y = hidden_states
            y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
            y = self.dropout(self.conv2(y).transpose(-1, 1))
            hidden_states = self.norm2(hidden_states + y)

        seq_out = self.fc_decoder_out(hidden_states)

        return seq_out


class S5Decoder(BaseDecoder):
    def __init__(
        self,
        neural_dim,
        n_classes,
        # preprocessing parameters
        nDays=24,
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        unfolding=True,
        input_nonlinearity="softsign",
        # processing parameters
        normalize_batch=False,
        init_embedding_layer=False,
        include_relu=False,
        fcc_layers=False,
        # model parameters
        d_model=256,
        ssm_size=384,
        n_layers=4, 
        blocks=1,
        bidirectional=False,
        conj_sym=False,
        dropout=0.1,
        prenorm=False,
        batchnorm=False,
        bn_momentum=0.95,
        bandlimit=None,
        step_rescale=1.0,
        clip_eigs=False,
        activation="gelu",
        add_embedding_layer=True,
        init_output_decoder_layer=False,
        add_residual=True,
        norm_epsilon=1e-5,
        rms_norm=False,
        C_init="lecun_normal",
    ):
        super(S5Decoder, self).__init__(
            neural_dim=neural_dim,
            nDays=nDays,
            strideLen=strideLen,
            kernelLen=kernelLen,
            gaussianSmoothWidth=gaussianSmoothWidth,
            unfolding=unfolding,
            input_nonlinearity=input_nonlinearity,
            add_embedding_layer=add_embedding_layer,
            init_embedding_layer=init_embedding_layer,
            embedding_layer_dim=d_model,
        )
        self.normalize_batch = normalize_batch
        self.fcc_layers = fcc_layers
        self.n_layers = n_layers
        self.include_relu = include_relu
        self.add_residual = add_residual
        self.dropout = nn.Dropout(p=dropout)

        # Block of model layers
        self.backbone = S5Model(
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

        self.fc_decoder_out = OutputDecoder(d_model, n_classes + 1, init_output_decoder_layer)  # +1 for CTC blank

        self.norm_f = (nn.LayerNorm if not rms_norm else torch.nn.RMSNorm)(d_model, eps=norm_epsilon)

        if self.fcc_layers:
            # TODO match with mamba layernorm
            d_ff = d_model
            self.activation = F.relu if activation == 'relu' else F.gelu
            self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
            self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

    
    def forward(self, neuralInput: torch.Tensor, dayIdx: torch.Tensor, states: Optional[List[torch.Tensor]] = None):
        """
        Forward pass of the Decoder
        Args:
            neuralInput: (batch_size x seq_len x num_features)
            dayIdx: (batch_size, )

        Returns:

        """
        if self.normalize_batch:
            dim_ = 1
            means = neuralInput.mean(dim_, keepdim=True).detach() # B x 1 x D
            neuralInput = neuralInput - means
            stdev = torch.sqrt(torch.var(neuralInput, dim=dim_, keepdim=True, unbiased=False) + 1e-5)  # B x 1 x D
            neuralInput /= stdev

        # Preprocess batch
        hidden_states = self.forward_preprocessing(neuralInput, dayIdx)

        # include relu
        if self.include_relu:
            hidden_states = torch.relu(hidden_states)
    

        if states is None:
            states = self.initial_state(hidden_states, hidden_states.shape[0])

        residual = hidden_states
        hidden_states, states = self.backbone(hidden_states, states)

        # Pass through FCC layers:
        if self.fcc_layers:
            y = hidden_states
            y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
            y = self.dropout(self.conv2(y).transpose(-1, 1))
            hidden_states = hidden_states + y

        if self.add_residual:
            hidden_states = hidden_states + residual
        hidden_states = self.norm_f(hidden_states)


        seq_out = self.fc_decoder_out(hidden_states)

        return seq_out

    @torch.no_grad()
    def initial_state(self, x, batch_size):
        model = self.backbone.layers[0].seq
        B_tilde, _ = model.seq.get_BC_tilde()
        Lambda = model.seq.get_lambda()
        step = torch.exp(model.seq.log_step[..., 0])
        Lambda_bar, B_bar = model.seq.discretize(Lambda, B_tilde, step)

        # Use last `kernelLen` inputs
        x_snippet = x[:, -self.kernelLen:].to(B_bar.dtype)  # shape: [B, kernelLen, D]

        state = torch.zeros(x.shape[0], Lambda_bar.shape[0], dtype=Lambda_bar.dtype, device=x.device)
        for i in range(self.kernelLen):
            u_ = x_snippet[:, i]  # [B, D]
            Bu = torch.einsum('pd,bd->bp', B_bar, u_)  # [B, P]
            state = Lambda_bar * state + Bu

        # For each layer, we need to initialize the state 
        states = [state.clone() for _ in range(self.n_layers)]
        return states

class SashimiDecoder(BaseDecoder):
    def __init__(
        self,
        neural_dim,
        n_classes,
        d_model,
        d_state,
        d_conv,
        expand_factor,
        layer_dim,
        nDays=24,
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional=False,
        bidirectional_input=False,
        unfolding=True,
        ssm_type='s4',  # Add SSM type parameter for Sashimi compatibility
        unet=False,  # Add UNet parameter for Sashimi compatibility
        pool=[4, 4],  # Add pooling configuration for Sashimi
        ff=2,  # Add feed-forward expansion factor
        dropout=0.0, # Add dropout
        input_nonlinearity="softsign",
        init_output_decoder_layer=False,
        add_embedding_layer=False,
        init_embedding_layer=False,
        embedding_layer_dim=None,
    ):
        super(SashimiDecoder, self).__init__(
            neural_dim=neural_dim,
            nDays=nDays,
            strideLen=strideLen,
            kernelLen=kernelLen,
            gaussianSmoothWidth=gaussianSmoothWidth,
            unfolding=unfolding,
            input_nonlinearity=input_nonlinearity,
            add_embedding_layer=add_embedding_layer,
            init_embedding_layer=init_embedding_layer,
            embedding_layer_dim=embedding_layer_dim,
        )
        self.layer_dim = layer_dim
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand_factor = expand_factor
        self.bidirectional_input = bidirectional_input

        if unfolding:
          input_dims = self.neural_dim * kernelLen
        else:
          input_dims = self.neural_dim

        # Linear transformation for the input to match the Sashimi model dimensions
        d_mamba = d_model * 2 if self.bidirectional_input else d_model

        d_out =  input_dims * (2 if self.bidirectional_input else 1)
        self.linear_input = OutputDecoder(
          d_out, d_mamba, init_output_decoder_layer=init_output_decoder_layer
        )

        # Initialize the Sashimi backbone with the necessary parameters
        self.sashimi = Sashimi(
          d_model=d_model,
          n_layers=layer_dim,
          pool=pool,
          expand=expand_factor,
          ff=ff,
          bidirectional=bidirectional,
          unet=unet,
          dropout=dropout,
          ssm_type=ssm_type,
          transposed=False
        )

        # Final layer to project the Sashimi output to the number of classes + CTC blank
        self.fc_decoder_out = OutputDecoder(self.d_model, n_classes + 1)

    def forward(self, neuralInput, dayIdx):
        # Preprocess the input (e.g., Gaussian smoothing and unfolding)
        stridedInputs = self.forward_preprocessing(neuralInput, dayIdx)

        if self.bidirectional_input:
            stridedFlip = torch.flip(stridedInputs, dims=(1,))
            stridedInputs = torch.cat((stridedInputs, stridedFlip), dim=-1)

        # Transform the input to match the Sashimi model input dimensions
        sashimi_in = self.linear_input(stridedInputs)

        # Pass through the Sashimi backbone
        hidden_states, _ = self.sashimi(sashimi_in)

        # Project the hidden states to the number of classes
        seq_out = self.fc_decoder_out(hidden_states)

        return seq_out


class BiTDecoder(nn.Module):
    def __init__(
        self,
        n_classes,
        # BiT specific parameters
        patch_size=4,
        dim=512,
        depth=8,
        heads=8,
        mlp_dim_ratio=4,
        dim_head=64,
        dropout=0.1,
        input_dropout=0.1,
        look_ahead=0,
        T5_style_pos=True,
        max_mask_pct=0.0,
        num_masks=1,
        mask_token_zeros=True,
        num_masks_channels=0,
        max_mask_channels=0,
        dist_dict_path=None,
        linderman_lab=False,
        # Base decoder parameters
        gaussianSmoothWidth=0,
        # get rid of this
        nDays=None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim_ratio = mlp_dim_ratio
        self.dim_head = dim_head
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.look_ahead = look_ahead
        self.T5_style_pos = T5_style_pos
        self.max_mask_pct = max_mask_pct
        self.num_masks = num_masks
        self.mask_token_zeros = mask_token_zeros
        self.num_masks_channels = num_masks_channels
        self.max_mask_channels = max_mask_channels
        self.dist_dict_path = dist_dict_path
        self.linderman_lab = linderman_lab

        # some classes 
        self.unfolding = False
        
        # Initialize the BiT model 
        # Note: patch_size here should match the neural_dim for proper channel handling
        self.bit_model = BiT_Phoneme(
            patch_size=patch_size,  # (time_patch, feature_patch) 
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim_ratio=mlp_dim_ratio,
            dim_head=dim_head,
            dropout=dropout,
            input_dropout=input_dropout,
            look_ahead=look_ahead,
            gaussianSmoothWidth=gaussianSmoothWidth,  # Use the configured value
            nClasses=n_classes,
            T5_style_pos=T5_style_pos,
            max_mask_pct=max_mask_pct,
            num_masks=num_masks,
            mask_token_zeros=mask_token_zeros,
            num_masks_channels=num_masks_channels,
            max_mask_channels=max_mask_channels,
            dist_dict_path=dist_dict_path,
            linderman_lab=linderman_lab,
        )

    def forward(self, neuralInput, X_len, dayIdx):
        # BiT model expects raw neural input and handles its own preprocessing
        # We apply minimal day-specific preprocessing but skip the unfolding
        
        # Apply day-specific transformations (weights and bias)
        batch_size, seq_len, num_features = neuralInput.shape
        # Select the weight matrix for the current day based on dayIdx
        #dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)  # (BS, N, N)
        
        # Apply a linear transformation to the neural input using the selected day weight matrix
        #transformedNeural = torch.einsum(
        #    "btd,bdk->btk", neuralInput, dayWeights
        #) + torch.index_select(self.dayBias, 0, dayIdx)  # (BS, L, N)
        
        # Apply input nonlinearity
        #if self.input_nonlinearity == "softsign":
        #    transformedNeural = self.inputLayerNonlinearity(transformedNeural)
        
        # BiT model expects X_len for length computation
        #X_len = torch.full((batch_size,), seq_len, device=neuralInput.device, dtype=torch.long)
        
        # Forward through BiT model - BiT expects (B, T, F) format
        output = self.bit_model(neuralInput, X_len, None)
        return output


class BiTMambaDecoder(nn.Module):
    def __init__(
        self,
        n_classes,
        # BiT specific parameters
        patch_size=4,
        dim=512,
        d_model=None,  # For mamba
        d_state=16,
        d_conv=4,
        expand_factor=1,
        layer_dim=8,  # replaces depth
        bidirectional=True,
        bidirectional_strategy="add",
        mamba_bi_new=True,
        dropout=0.1,
        input_dropout=0.1,
        look_ahead=0,
        T5_style_pos=True,
        max_mask_pct=0.0,
        num_masks=1,
        mask_token_zeros=True,
        num_masks_channels=0,
        max_mask_channels=0,
        dist_dict_path=None,
        linderman_lab=False,
        # Base decoder parameters
        gaussianSmoothWidth=0,
        # Additional parameters that might be passed from config
        depth=None,  # For compatibility 
        heads=6,
        mlp_dim_ratio=4,
        dim_head=64,
        # get rid of this
        nDays=None,
    ):
        super().__init__()
        
        # Use dim as d_model if not specified
        if d_model is None:
            d_model = dim
            
        # Use depth as layer_dim if specified 
        if depth is not None:
            layer_dim = depth
            
        self.patch_size = patch_size
        self.dim = dim
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand_factor = expand_factor
        self.layer_dim = layer_dim
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        self.mamba_bi_new = mamba_bi_new
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.look_ahead = look_ahead
        self.T5_style_pos = T5_style_pos
        self.max_mask_pct = max_mask_pct
        self.num_masks = num_masks
        self.mask_token_zeros = mask_token_zeros
        self.num_masks_channels = num_masks_channels
        self.max_mask_channels = max_mask_channels
        self.dist_dict_path = dist_dict_path
        self.linderman_lab = linderman_lab
        self.depth = depth
        self.heads = heads
        self.mlp_dim_ratio = mlp_dim_ratio
        self.dim_head = dim_head

        # some classes 
        self.unfolding = False
        
        # Initialize the BiT-Mamba model 
        self.bit_model = BiT_Mamba_Phoneme(
            patch_size=patch_size,
            dim=dim,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
            layer_dim=layer_dim,
            bidirectional=bidirectional,
            bidirectional_strategy=bidirectional_strategy,
            mamba_bi_new=mamba_bi_new,
            dropout=dropout,
            input_dropout=input_dropout,
            look_ahead=look_ahead,
            gaussianSmoothWidth=gaussianSmoothWidth,
            nClasses=n_classes,
            T5_style_pos=T5_style_pos,
            max_mask_pct=max_mask_pct,
            num_masks=num_masks,
            mask_token_zeros=mask_token_zeros,
            num_masks_channels=num_masks_channels,
            max_mask_channels=max_mask_channels,
            dist_dict_path=dist_dict_path,
            linderman_lab=linderman_lab,
            # Add transformer params for compatibility 
            depth=layer_dim,
            heads=heads,
            mlp_dim_ratio=mlp_dim_ratio,
            dim_head=dim_head,
        )

    def forward(self, neuralInput, X_len, dayIdx):
        # Forward through BiT-Mamba model
        output = self.bit_model(neuralInput, X_len, None)
        return output


class BiTS5Decoder(nn.Module):
    def __init__(
        self,
        n_classes,
        # BiT specific parameters
        patch_size=4,
        dim=512,
        d_model=None,  # For S5
        d_state=16,
        n_layers=8,  # replaces depth
        dropout=0.1,
        input_dropout=0.1,
        look_ahead=0,
        T5_style_pos=True,
        max_mask_pct=0.0,
        num_masks=1,
        mask_token_zeros=True,
        num_masks_channels=0,
        max_mask_channels=0,
        dist_dict_path=None,
        linderman_lab=False,
        # Base decoder parameters
        gaussianSmoothWidth=0,
        # Additional parameters that might be passed from config
        depth=None,  # For compatibility 
        heads=6,
        mlp_dim_ratio=4,
        dim_head=64,
        # get rid of this
        nDays=None,
    ):
        super().__init__()
        
        # Use dim as d_model if not specified
        if d_model is None:
            d_model = dim
            
        # Use depth as n_layers if specified 
        if depth is not None:
            n_layers = depth
            
        self.patch_size = patch_size
        self.dim = dim
        self.d_model = d_model
        self.d_state = d_state
        self.n_layers = n_layers
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.look_ahead = look_ahead
        self.T5_style_pos = T5_style_pos
        self.max_mask_pct = max_mask_pct
        self.num_masks = num_masks
        self.mask_token_zeros = mask_token_zeros
        self.num_masks_channels = num_masks_channels
        self.max_mask_channels = max_mask_channels
        self.dist_dict_path = dist_dict_path
        self.linderman_lab = linderman_lab
        self.depth = depth
        self.heads = heads
        self.mlp_dim_ratio = mlp_dim_ratio
        self.dim_head = dim_head

        # some classes 
        self.unfolding = False
        
        # Initialize the BiT-S5 model 
        self.bit_model = BiT_S5_Phoneme(
            patch_size=patch_size,
            dim=dim,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            dropout=dropout,
            input_dropout=input_dropout,
            look_ahead=look_ahead,
            gaussianSmoothWidth=gaussianSmoothWidth,
            nClasses=n_classes,
            T5_style_pos=T5_style_pos,
            max_mask_pct=max_mask_pct,
            num_masks=num_masks,
            mask_token_zeros=mask_token_zeros,
            num_masks_channels=num_masks_channels,
            max_mask_channels=max_mask_channels,
            dist_dict_path=dist_dict_path,
            linderman_lab=linderman_lab,
            # Add transformer params for compatibility 
            depth=n_layers,
            heads=heads,
            mlp_dim_ratio=mlp_dim_ratio,
            dim_head=dim_head,
        )

    def forward(self, neuralInput, X_len, dayIdx):
        # Forward through BiT-S5 model
        output = self.bit_model(neuralInput, X_len, None)
        return output


class BiTMamba2Decoder(nn.Module):
    def __init__(
        self,
        n_classes,
        # BiT specific parameters
        patch_size=4,
        dim=512,
        d_model=None,  # For mamba2
        d_state=16,
        d_conv=4,
        expand_factor=1,
        layer_dim=8,  # replaces depth
        dropout=0.1,
        input_dropout=0.1,
        look_ahead=0,
        T5_style_pos=True,
        max_mask_pct=0.0,
        num_masks=1,
        mask_token_zeros=True,
        num_masks_channels=0,
        max_mask_channels=0,
        dist_dict_path=None,
        linderman_lab=False,
        # Base decoder parameters
        gaussianSmoothWidth=0,
        # Additional parameters that might be passed from config
        depth=None,  # For compatibility 
        heads=6,
        mlp_dim_ratio=4,
        dim_head=64,
        # get rid of this
        nDays=None,
    ):
        super().__init__()
        
        # Use dim as d_model if not specified
        if d_model is None:
            d_model = dim
            
        # Use depth as layer_dim if specified 
        if depth is not None:
            layer_dim = depth
            
        self.patch_size = patch_size
        self.dim = dim
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand_factor = expand_factor
        self.layer_dim = layer_dim
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.look_ahead = look_ahead
        self.T5_style_pos = T5_style_pos
        self.max_mask_pct = max_mask_pct
        self.num_masks = num_masks
        self.mask_token_zeros = mask_token_zeros
        self.num_masks_channels = num_masks_channels
        self.max_mask_channels = max_mask_channels
        self.dist_dict_path = dist_dict_path
        self.linderman_lab = linderman_lab
        self.depth = depth
        self.heads = heads
        self.mlp_dim_ratio = mlp_dim_ratio
        self.dim_head = dim_head

        # some classes 
        self.unfolding = False
        
        # Initialize the BiT-Mamba2 model 
        self.bit_model = BiT_Mamba2_Phoneme(
            patch_size=patch_size,
            dim=dim,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
            layer_dim=layer_dim,
            dropout=dropout,
            input_dropout=input_dropout,
            look_ahead=look_ahead,
            gaussianSmoothWidth=gaussianSmoothWidth,
            nClasses=n_classes,
            T5_style_pos=T5_style_pos,
            max_mask_pct=max_mask_pct,
            num_masks=num_masks,
            mask_token_zeros=mask_token_zeros,
            num_masks_channels=num_masks_channels,
            max_mask_channels=max_mask_channels,
            dist_dict_path=dist_dict_path,
            linderman_lab=linderman_lab,
            # Add transformer params for compatibility 
            depth=layer_dim,
            heads=heads,
            mlp_dim_ratio=mlp_dim_ratio,
            dim_head=dim_head,
        )

    def forward(self, neuralInput, X_len, dayIdx):
        # Forward through BiT-Mamba2 model
        output = self.bit_model(neuralInput, X_len, None)
        return output

