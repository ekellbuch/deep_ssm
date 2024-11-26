from deep_ssm.data.data_transforms import GaussianSmoothing
import torch
import torch.nn as nn
# from deep_ssm.mixers.mamba_extra import MixerModel
# from deep_ssm.models.audio_models import Sashimi
import torch.nn.functional as F

import numpy as np
from accelerated_scan.warp import (
    scan,
)  # from https://github.com/proger/accelerated-scan


def quasi_deer_torch(
    f,
    diagonal_derivative,
    initial_state,  # (B,D)
    states_guess,  # (B,D, T)
    drivers,  # (B, d_input)
    num_iters=10,  # controls number of newton iterations
    k=0.0,  # controls the amount of damping
):
    """
    Currently is quasi-DEER/ELK

    Args:
      f: a forward fxn that takes in a full state and an input, and outputs the next full state.
          In the context of a GRU, f is a GRU cell, the full state is the hidden state, and the driver is the input
          In pytorch setting, f should be able to handle the batch dimension
      diagonal_derivative: a forward fxn that takes in full state and an input, and returns a length D diagonal derivative
          In pytorch setting, f should be able to handle the batch dimension
      initial_state: (B,D)
      states_guess, (B, D, T-1)
      drivers, jax.Array, (B, d_input, T-1)
      num_iters: number of iterations to run
      k: int, k=0 is quasi-DEER, k is between 0 and 1, nonzero k is slim-quasi-ELK
    Notes:
    - The initial_state is NOT the same as the initial mean we give to dynamax
    - The initial_mean is something on which we do inference
    - The initial_state is the fixed starting point.

    The structure looks like the following.
    Let h0 be the initial_state (fixed), h[1:T-1] be the states, and e[0:T-2] be the drivers

    Then our graph looks like

    h0 -----> h1 ---> h2 ---> ..... h_{T-2} ----> h_{T-1}
              |       |                   |          |
              e1      e2       ..... e_{T-2}      e_{T-1}

    Use the pytorch scan from: https://github.com/proger/accelerated-scan
    This scan expects inputs in the form (B,D,T)
    Note that the RNN standard (for inputs) in pytorch is (T,B,d_input)

    This scan also requires powers of 2, so padding for now...
    """
    B = states_guess.shape[0]
    D = states_guess.shape[1]
    T = states_guess.shape[-1]
    padded_T = int(2 ** np.ceil(np.log2(T)))  # must be a power of 2
    device = states_guess.device

    def step(states):
        """
        states: B,D,T
        """
        # Evaluate f and its Jacobian in parallel across timesteps 1,..,T-1
        fs = torch.func.vmap(f, in_dims=-1, out_dims=-1)(
            states[..., :-1], drivers[..., 1:]
        )  # (B,D,T-1)

        # Compute the As and bs from fs and Jfs
        As = (1.0 - k) * torch.func.vmap(diagonal_derivative, in_dims=-1, out_dims=-1)(
            states[..., :-1], drivers[..., 1:]
        )  # (B, D, T-1)
        bs = fs - As * states[..., :-1]  # (B, D, T-1)

        # initial_state is h0
        b0 = f(initial_state, drivers[..., 0])  # h1, (B,D)
        A0 = torch.zeros_like(As[..., 0])  # (B,D)
        A = torch.cat(
            [A0.unsqueeze(-1), As, torch.ones([B, D, padded_T - T], device=device)],
            dim=-1,
        )  # (B, D, T)
        b = torch.cat(
            [b0.unsqueeze(-1), bs, torch.zeros([B, D, padded_T - T], device=device)],
            dim=-1,
        )  # (B, D, T)

        # run appropriate parallel alg
        new_states = scan(A, b)[..., :T]  # (B, D, T)
        # trying in place modification to be more memory efficient
        new_states.nan_to_num() # zero out nans
        # new_states = torch.nan_to_num(new_states)  # zero out nans
        return new_states

    deer_traces = []
    for i in range(num_iters):
        states_guess = step(states_guess)
        deer_traces.append(states_guess)

    return deer_traces[-1] # (B, D, T)


class MinRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MinRNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.input_weights = nn.Parameter(
            torch.randn(hidden_size, input_size) / (input_size**0.5)
        )
        self.input_bias = nn.Parameter(torch.zeros(hidden_size))
        self.recurrent_weights = nn.Parameter(
            torch.randn(hidden_size, hidden_size) / (hidden_size**0.5)
        )
        self.U_z = nn.Parameter(
            torch.randn(hidden_size, hidden_size) / (hidden_size**0.5)
        )
        self.b_u = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, prev_state, input):
        z = torch.tanh(torch.matmul(input, self.input_weights.T) + self.input_bias)
        u = torch.sigmoid(
            torch.matmul(prev_state, self.recurrent_weights.T)
            + torch.matmul(z, self.U_z.T)
            + self.b_u
        )
        state = u * prev_state + (1 - u) * z
        return state

    def diagonal_derivative(self, prev_state, input):
        z = torch.tanh(torch.matmul(input, self.input_weights.T) + self.input_bias)
        u = torch.sigmoid(
            torch.matmul(prev_state, self.recurrent_weights.T)
            + torch.matmul(z, self.U_z.T)
            + self.b_u
        )
        derivative = (
            u + (prev_state - z) * u * (1 - u) * self.recurrent_weights.diagonal()
        )
        return derivative


class MinRNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=True,
        dropout=0,
        bidirectional=True,
        num_iters=2, # number of iterations for quasi-DEER
    ):
        super(MinRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional

        num_directions = 2 if bidirectional else 1
        self.num_directions = num_directions

        self.forward_cells = nn.ModuleList(
            [
                MinRNNCell(
                    input_size if i == 0 else hidden_size * num_directions, hidden_size
                )
                for i in range(num_layers)
            ]
        )
        if bidirectional:
            self.backward_cells = nn.ModuleList(
                [
                    MinRNNCell(
                        input_size if i == 0 else hidden_size * num_directions,
                        hidden_size,
                    )
                    for i in range(num_layers)
                ]
            )

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        self.num_iters = num_iters # number of iterations for quasi-DEER

    def forward(self, input, hx=None):
        if not self.batch_first:
            input = input.transpose(0, 1)

        batch_size, seq_len, _ = input.size()
        num_directions = 2 if self.bidirectional else 1

        if hx is None:
            hx = torch.zeros(
                self.num_layers * num_directions,
                batch_size,
                self.hidden_size,
                device=input.device,
            )

        output = input
        h_n = []

        for layer in range(self.num_layers):
            forward_layer_output = self._process_layer(
                output, hx[layer * num_directions], self.forward_cells[layer]
            )
            if self.bidirectional:
                backward_layer_output = self._process_layer(
                    output.flip(1),
                    hx[layer * num_directions + 1],
                    self.backward_cells[layer],
                )
                backward_layer_output = backward_layer_output.flip(1)
                layer_output = torch.cat(
                    [forward_layer_output, backward_layer_output], dim=2
                )
            else:
                layer_output = forward_layer_output

            h_n.append(layer_output[:, -1, : self.hidden_size])
            if self.bidirectional:
                h_n.append(layer_output[:, 0, self.hidden_size :])

            output = layer_output
            if self.dropout_layer and layer < self.num_layers - 1:
                output = self.dropout_layer(output)

        h_n = torch.stack(h_n)

        if not self.batch_first:
            output = output.transpose(0, 1)

        return output, h_n

    def _process_layer(self, input, h0, cell):
        batch_size, seq_len, _ = input.size()
        h0 = h0.unsqueeze(1).expand(-1, seq_len, -1)
        input = input.transpose(1, 2)  # (batch_size, hidden_size, seq_len)

        device = input.device
        states_guess = torch.zeros(batch_size, self.hidden_size, seq_len, device=device)
        output = quasi_deer_torch(
            f=cell,
            diagonal_derivative=cell.diagonal_derivative,
            initial_state=h0[:, 0],
            states_guess=states_guess,
            drivers=input,
            num_iters=self.num_iters,
        ) # (B,D,T)

        return output.transpose(1, 2)  # (batch_size, seq_len, hidden_size)


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


class MinRNNDecoder(BaseDecoder):
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
        num_iters=2, # number of iterations for quasi-DEER
    ):
        super(MinRNNDecoder, self).__init__(
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

        self.minrnn_decoder = MinRNN(
            input_size=input_dims,
            hidden_size=hidden_dim,
            num_layers=layer_dim,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
            num_iters=num_iters,
        )

        # for name, param in self.gru_decoder.named_parameters():
        #     if "weight_hh" in name:
        #         nn.init.orthogonal_(param)
        #     if "weight_ih" in name:
        #         nn.init.xavier_uniform_(param)

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
        hid, _ = self.minrnn_decoder(stridedInputs, h0.detach())

        # Apply Decoder
        seq_out = self.fc_decoder_out(hid)
        return seq_out


# class MambaDecoder(BaseDecoder):
#     def __init__(
#         self,
#         neural_dim,
#         n_classes,
#         d_model,
#         d_state,
#         d_conv,
#         expand_factor,
#         layer_dim,
#         nDays=24,
#         strideLen=4,
#         kernelLen=14,
#         gaussianSmoothWidth=0,
#         bidirectional_input=False,
#         bidirectional=False,
#         unfolding=True,
#         mamba_bi_new=True,
#         input_nonlinearity="softsign",
#         fused_add_norm=False,
#         rms_norm=False,
#         initialize_mixer=False,
#         bidirectional_strategy=None,
#         dropout=0.0,
#         normalize_batch=False,
#         init_embedding_layer=False,
#         include_relu=False,
#         # additional decoding layer:
#         fcc_layers=False,
#         activation="relu",
#         dff=None,
#     ):
#         super(MambaDecoder, self).__init__(
#             neural_dim=neural_dim,
#             nDays=nDays,
#             strideLen=strideLen,
#             kernelLen=kernelLen,
#             gaussianSmoothWidth=gaussianSmoothWidth,
#             unfolding=unfolding,
#             input_nonlinearity=input_nonlinearity,
#         )
#         self.layer_dim = layer_dim
#         self.d_model = d_model
#         self.d_state = d_state
#         self.d_conv = d_conv
#         self.expand_factor = expand_factor
#         self.normalize_batch = normalize_batch
#         self.include_relu = include_relu
#         self.fcc_layers = fcc_layers

#         if bidirectional_input:
#             raise NotImplementedError("Bidirectional input not supported for MambaDecoder")

#         if unfolding:
#           input_dims = self.neural_dim * kernelLen
#         else:
#           input_dims = self.neural_dim

#         # input dimension to model dimension
#         self.linear_input = nn.Linear(input_dims, d_model)
#         self.dropout = nn.Dropout(p=dropout)

#         # Block of model layers
#         self.backbone = MixerModel(
#             d_model=d_model,
#             d_state=d_state,
#             d_conv=d_conv,
#             expand_factor=expand_factor,
#             n_layer=layer_dim,
#             rms_norm=rms_norm,
#             fused_add_norm=fused_add_norm,
#             bidirectional=bidirectional,
#             mamba_bi_new=mamba_bi_new,
#             initialize_mixer=initialize_mixer,
#             bidirectional_strategy=bidirectional_strategy,
#         )

#         # from model dimension to n_classes
#         if bidirectional and bidirectional_strategy == "concatenate":
#             d_output = d_model*2
#         else:
#             d_output = d_model
#         self.fc_decoder_out = nn.Linear(d_output, n_classes + 1)  # +1 for CTC blank

#         # Initialize embedding weights:
#         if init_embedding_layer:
#             for layer in [self.linear_input, self.fc_decoder_out]:
#                 nn.init.xavier_uniform_(layer.weight)
#                 nn.init.zeros_(layer.bias)

#         if self.fcc_layers:
#             d_ff = dff or 4*d_output
#             self.activation = F.relu if activation == 'relu' else F.gelu
#             self.conv1 = nn.Conv1d(in_channels=d_output, out_channels=d_ff, kernel_size=1)
#             self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_output, kernel_size=1)
#             self.norm2 = nn.LayerNorm(d_output)


#     def forward(self, neuralInput, dayIdx):
#         """
#         Forward pass of the Decoder
#         Args:
#             neuralInput: (batch_size x seq_len x num_features)
#             dayIdx: (batch_size, )

#         Returns:

#         """
#         if self.normalize_batch:
#             dim_ = 1
#             means = neuralInput.mean(dim_, keepdim=True).detach() # B x 1 x D
#             neuralInput = neuralInput - means
#             stdev = torch.sqrt(torch.var(neuralInput, dim=dim_, keepdim=True, unbiased=False) + 1e-5)  # B x 1 x D
#             neuralInput /= stdev

#         # Preprocess batch
#         stridedInputs = self.forward_preprocessing(neuralInput, dayIdx)

#         hidden_states = self.linear_input(stridedInputs)
#         # include relu
#         if self.include_relu:
#             hidden_states = torch.relu(hidden_states)

#         hidden_states = self.dropout(hidden_states)

#         # Pass through the mixer
#         hidden_states = self.backbone(hidden_states)

#         # Pass through FCC layers:
#         if self.fcc_layers:
#             y = hidden_states
#             y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
#             y = self.dropout(self.conv2(y).transpose(-1, 1))
#             hidden_states = self.norm2(hidden_states + y)

#         seq_out = self.fc_decoder_out(hidden_states)

#         return seq_out


# class SashimiDecoder(BaseDecoder):
#     def __init__(
#         self,
#         neural_dim,
#         n_classes,
#         d_model,
#         d_state,
#         d_conv,
#         expand_factor,
#         layer_dim,
#         nDays=24,
#         strideLen=4,
#         kernelLen=14,
#         gaussianSmoothWidth=0,
#         bidirectional=False,
#         bidirectional_input=False,
#         unfolding=True,
#         ssm_type='s4',  # Add SSM type parameter for Sashimi compatibility
#         unet=False,  # Add UNet parameter for Sashimi compatibility
#         pool=[4, 4],  # Add pooling configuration for Sashimi
#         ff=2,  # Add feed-forward expansion factor
#         dropout=0.0, # Add dropout
#         input_nonlinearity="softsign",
#     ):
#         super(SashimiDecoder, self).__init__(
#             neural_dim=neural_dim,
#             nDays=nDays,
#             strideLen=strideLen,
#             kernelLen=kernelLen,
#             gaussianSmoothWidth=gaussianSmoothWidth,
#             unfolding=unfolding,
#             input_nonlinearity=input_nonlinearity,
#         )
#         self.layer_dim = layer_dim
#         self.d_model = d_model
#         self.d_state = d_state
#         self.d_conv = d_conv
#         self.expand_factor = expand_factor
#         self.bidirectional_input = bidirectional_input

#         if unfolding:
#           input_dims = self.neural_dim * kernelLen
#         else:
#           input_dims = self.neural_dim

#         # Linear transformation for the input to match the Sashimi model dimensions
#         d_mamba = d_model * 2 if self.bidirectional_input else d_model

#         self.linear_input = nn.Linear(
#           input_dims * (2 if self.bidirectional_input else 1), d_mamba
#         )

#         # Initialize the Sashimi backbone with the necessary parameters
#         self.sashimi = Sashimi(
#           d_model=d_model,
#           n_layers=layer_dim,
#           pool=pool,
#           expand=expand_factor,
#           ff=ff,
#           bidirectional=bidirectional,
#           unet=unet,
#           dropout=dropout,
#           ssm_type=ssm_type,
#           transposed=False
#         )

#         # Final layer to project the Sashimi output to the number of classes + CTC blank
#         self.fc_decoder_out = nn.Linear(self.d_model, n_classes + 1)

#     def forward(self, neuralInput, dayIdx):
#         # Preprocess the input (e.g., Gaussian smoothing and unfolding)
#         stridedInputs = self.forward_preprocessing(neuralInput, dayIdx)

#         if self.bidirectional_input:
#             stridedFlip = torch.flip(stridedInputs, dims=(1,))
#             stridedInputs = torch.cat((stridedInputs, stridedFlip), dim=-1)

#         # Transform the input to match the Sashimi model input dimensions
#         sashimi_in = self.linear_input(stridedInputs)

#         # Pass through the Sashimi backbone
#         hidden_states, _ = self.sashimi(sashimi_in)

#         # Project the hidden states to the number of classes
#         seq_out = self.fc_decoder_out(hidden_states)
#         return seq_out
