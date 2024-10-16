from deep_ssm.data.data_transforms import GaussianSmoothing
import torch
import torch.nn as nn
from deep_ssm.mixers.mamba_extra import MixerModel
from deep_ssm.models.audio_models import Sashimi


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
        d_model,
        d_state,
        d_conv,
        expand_factor,
        layer_dim,
        nDays=24,
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional_input=False,
        bidirectional=False,
        unfolding=True,
        mamba_bi_new=True,
        input_nonlinearity="softsign",
        fused_add_norm=False,
        rms_norm=False,
        initialize_mixer=False,
        bidirectional_strategy=None,
        dropout=0.0,
        normalize_batch=False,
        init_embedding_layer=False,
        include_relu=False
    ):
        super(MambaDecoder, self).__init__(
            neural_dim=neural_dim,
            nDays=nDays,
            strideLen=strideLen,
            kernelLen=kernelLen,
            gaussianSmoothWidth=gaussianSmoothWidth,
            unfolding=unfolding,
            input_nonlinearity=input_nonlinearity,
        )
        self.layer_dim = layer_dim
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand_factor = expand_factor
        self.normalize_batch = normalize_batch
        self.include_relu = include_relu

        if bidirectional_input:
            raise NotImplementedError("Bidirectional input not supported for MambaDecoder")

        if unfolding:
          input_dims = self.neural_dim * kernelLen
        else:
          input_dims = self.neural_dim

        # input dimension to model dimension
        self.linear_input = nn.Linear(input_dims, d_model)
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
        self.fc_decoder_out = nn.Linear(d_output, n_classes + 1)  # +1 for CTC blank

        # Initialize embedding weights:
        if init_embedding_layer:
            for layer in [self.linear_input, self.fc_decoder_out]:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

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
        stridedInputs = self.forward_preprocessing(neuralInput, dayIdx)

        hidden_states = self.linear_input(stridedInputs)
        # include relu
        if self.include_relu:
            hidden_states = torch.relu(hidden_states)
    
        hidden_states = self.dropout(hidden_states)

        # Pass through the mixer
        hidden_states = self.backbone(hidden_states)

        seq_out = self.fc_decoder_out(hidden_states)

        return seq_out


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
    ):
        super(SashimiDecoder, self).__init__(
            neural_dim=neural_dim,
            nDays=nDays,
            strideLen=strideLen,
            kernelLen=kernelLen,
            gaussianSmoothWidth=gaussianSmoothWidth,
            unfolding=unfolding,
            input_nonlinearity=input_nonlinearity,
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

        self.linear_input = nn.Linear(
          input_dims * (2 if self.bidirectional_input else 1), d_mamba
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
        self.fc_decoder_out = nn.Linear(self.d_model, n_classes + 1)

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

