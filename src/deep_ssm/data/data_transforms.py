import torch
import numbers
from torch import nn
import math
import torch.nn.functional as F

class Transform(object):
  """Abstract base class for transforms."""

  def __call__(self, *args):
    raise NotImplementedError

  def __repr__(self):
    raise NotImplementedError


class AddWhiteNoise(Transform):
    """Add indepdent noise to each neuron across time"""

    def __init__(self, whiteNoiseSD=0):
        self.whiteNoiseSD = whiteNoiseSD

    def __call__(self, X):
        X += torch.randn(X.shape, device=X.device) * self.whiteNoiseSD
        return X


class AddOffset(Transform):
  """Add offset to each neuron"""

  def __init__(self, constantOffsetSD=0):
    self.constantOffsetSD = constantOffsetSD

  def __call__(self, X):
    X += (torch.randn([1, X.shape[-1]], device=X.device) * self.constantOffsetSD)
    return X


class SpeckleMasking(Transform):
  def __init__(self, speckled_mask_p=0, mask_value=0, renormalize_masking=True):
    self.mask_p = speckled_mask_p
    self.mask_value = mask_value
    self.renormalize_masking = renormalize_masking

  def __call__(self, X):
    n_timesteps, n_features = X.shape
    n = n_timesteps * n_features

    maskt = torch.randint(low=0, high=n_timesteps, size=(int(n * self.mask_p),))
    maskf = torch.randint(low=0, high=n_features, size=(int(n * self.mask_p),))
    # Compute flattened indices for masking
    mask_idx = n_features * maskt + maskf

    # Flatten the input tensor and adjust for the masking probability
    inputs_masked_flattened = X.flatten()
    if self.renormalize_masking:
      inputs_masked_flattened = inputs_masked_flattened * 1 / (1 - self.mask_p)

    # Apply masking by setting the selected elements to the speckled_masking_value
    inputs_masked_flattened[mask_idx] = self.mask_value

    # Reshape the flattened tensor back to the original shape
    X = inputs_masked_flattened.reshape((n_timesteps, n_features))

    return X


class FeatureMasking(Transform):
  """
  Mask features in the input tensor:

  """
  def __init__(self, feature_mask_p=0, mask_value=0):
    self.feature_mask_p = feature_mask_p
    self.mask_value = mask_value

  def __call__(self, X):
    # TODO: (maybe) increase regularization by masking more features or for different chunks
    n_timesteps, n_features = X.shape

    num_masked_features = int(n_features * self.feature_mask_p)
    # features to be masked
    mask_f = torch.randint(low=0, high=n_features, size=(num_masked_features,))

    X[:, mask_f] = self.mask_value

    return X


class TemporalMasking(Transform):
  """
  Mask features in the input tensor:

  """
  def __init__(self, temporal_mask_n=0, mask_value=0, temporal_mask_len=2):
    self.temporal_mask_n = temporal_mask_n
    self.mask_value = mask_value
    self.temporal_mask_len = temporal_mask_len

  def __call__(self, X):
    # want to add n masks
    n_timesteps, n_features = X.shape

    mask_init = torch.randint(low=0, high=(n_timesteps- self.temporal_mask_len)+1, size=(self.temporal_mask_n,))
    mask_end = mask_init + self.temporal_mask_len
    X[mask_init:mask_end] = self.mask_value

    return X


class GaussianSmoothing(nn.Module):
  """
  Apply gaussian smoothing on a
  1d, 2d or 3d tensor. Filtering is performed seperately for each channel
  in the input using a depthwise convolution.
  Arguments:
      channels (int, sequence): Number of channels of the input tensors. Output will
          have this number of channels as well.
      kernel_size (int, sequence): Size of the gaussian kernel.
      sigma (float, sequence): Standard deviation of the gaussian kernel.
      dim (int, optional): The number of dimensions of the data.
          Default value is 2 (spatial).
  """

  def __init__(self, channels, kernel_size, sigma, dim=2):
    super(GaussianSmoothing, self).__init__()
    if isinstance(kernel_size, numbers.Number):
      kernel_size = [kernel_size] * dim
    if isinstance(sigma, numbers.Number):
      sigma = [sigma] * dim

    # The gaussian kernel is the product of the
    # gaussian function of each dimension.
    kernel = 1
    meshgrids = torch.meshgrid(
      [torch.arange(size, dtype=torch.float32) for size in kernel_size]
    )
    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
      mean = (size - 1) / 2
      kernel *= (
        1
        / (std * math.sqrt(2 * math.pi))
        * torch.exp(-(((mgrid - mean) / std) ** 2) / 2)
      )

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)

    # Reshape to depthwise convolutional weight
    kernel = kernel.view(1, 1, *kernel.size())
    kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

    self.register_buffer("weight", kernel)
    self.groups = channels

    if dim == 1:
      self.conv = F.conv1d
    elif dim == 2:
      self.conv = F.conv2d
    elif dim == 3:
      self.conv = F.conv3d
    else:
      raise RuntimeError(
        "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
      )

  def forward(self, input):
    """
    Apply gaussian filter to input.
    Arguments:
        input (torch.Tensor): Input to apply gaussian filter on.
    Returns:
        filtered (torch.Tensor): Filtered output.
    """
    return self.conv(input, weight=self.weight, groups=self.groups, padding="same")
