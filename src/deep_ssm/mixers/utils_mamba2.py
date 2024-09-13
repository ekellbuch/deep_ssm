"""
Separate python only code for compatibility without triton
"""
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from deep_ssm.mixers.utils_mamba import causal_conv1d_ref as causal_conv1d_fn


def rms_norm_ref(x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True, upcast=True):
  dtype = x.dtype
  N = x.shape[-1]
  weight = weight.float()
  bias = bias.float() if bias is not None else None
  if upcast:
    x = x.float()
    z = z.float() if z is not None else z
  if z is not None and not norm_before_gate:
    x = x * F.silu(z)
  if group_size is None:
    rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
    out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
  else:
    x_group = rearrange(x, "... (g d) -> ... g d", d=group_size)
    rstd = 1 / torch.sqrt((x_group.square()).mean(dim=-1, keepdim=True) + eps)
    out = rearrange(x_group * rstd, "... g d -> ... (g d)") * weight
    if bias is not None:
      out = out + bias
  if z is not None and norm_before_gate:
    out *= F.silu(z)
  return out.to(dtype)



def state_passing_ref(states, dA_chunk_cumsum, initial_states=None):
  """
  Argument:
      states: (batch, nchunks, nheads, dim)
      dA_chunk_cumsum: (batch, nheads, nchunks)
      initial_states: (batch, nheads, dim)
  Return:
      out: (batch, nchunks, nheads, dim)
      final_states: (batch, nheads, dim)
  """
  if initial_states is None:
    initial_states = torch.zeros_like(states[:, 0])
  states = torch.cat([rearrange(initial_states, "b h d -> b 1 h d"), states], dim=1)
  dA_chunk_cumsum = F.pad(dA_chunk_cumsum, (1, 0))
  dA_chunk_cumsum = torch.cumsum(dA_chunk_cumsum, dim=-1)
  nchunks = dA_chunk_cumsum.shape[-1]
  # (batch, nheads, nchunks, nchunks)
  dt_chunk_segment_sum = dA_chunk_cumsum[:, :, :, None] - dA_chunk_cumsum[:, :, None, :]
  # (batch, nheads, nchunks, nchunks)
  decay_chunk = torch.exp(dt_chunk_segment_sum)
  causal_mask = torch.tril(torch.ones(nchunks, nchunks, device=states.device, dtype=bool), diagonal=0)
  decay_chunk = decay_chunk.masked_fill(~causal_mask, 0)
  out = torch.einsum("bhzc,bchd->bzhd", decay_chunk.to(dtype=states.dtype), states)
  return out[:, :-1], out[:, -1]

def chunk_state_ref(B, x, dt, dA_cumsum):
  """
  Argument:
      B: (batch, seqlen, ngroups, headdim)
      x: (batch, seqlen, nheads, headdim)
      dt: (batch, nheads, nchunks, chunk_size)
      dA_cumsum: (batch, nheads, nchunks, chunk_size)
  Return:
      states: (batch, nchunks, nheads, headdim, dstate)
  """
  # Check constraints.
  batch, seqlen, nheads, headdim = x.shape
  dstate = B.shape[-1]
  _, _, nchunks, chunk_size = dt.shape
  assert seqlen <= nchunks * chunk_size
  assert x.shape == (batch, seqlen, nheads, headdim)
  assert dt.shape == (batch, nheads, nchunks, chunk_size)
  ngroups = B.shape[2]
  assert nheads % ngroups == 0
  assert B.shape == (batch, seqlen, ngroups, dstate)
  B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
  assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
  if seqlen < nchunks * chunk_size:
    x = F.pad(x, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
    B = F.pad(B, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
  x = rearrange(x, "b (c l) h p -> b c l h p", l=chunk_size)
  B = rearrange(B, "b (c l) ... -> b c l ...", l=chunk_size)
  decay_states = torch.exp((dA_cumsum[:, :, :, -1:] - dA_cumsum))
  return torch.einsum("bclhn,bhcl,bhcl,bclhp->bchpn", B.to(x.dtype), decay_states.to(x.dtype), dt.to(x.dtype), x)



def chunk_scan_ref(B, C, x, dt, dA_cumsum, prev_states, D=None, z=None):
    """
    Argument:
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
        prev_states: (batch, nchunks, nheads, headdim, dstate)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    assert B.shape == (batch, seqlen, ngroups, dstate)
    _, _, nchunks, chunk_size = dt.shape
    assert seqlen == nchunks * chunk_size
    assert C.shape == B.shape
    B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
    C = repeat(C, "b l g d -> b l (g h) d", h=nheads // ngroups)
    CB = torch.einsum("bclhn,bcshn->bchls", rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
                      rearrange(B, "b (c s) h n -> b c s h n", c=nchunks))
    # (batch, nheads, nchunks, chunksize, chunksize)
    dt_segment_sum = dA_cumsum[:, :, :, :, None] - dA_cumsum[:, :, :, None, :]
    decay = torch.exp(dt_segment_sum)
    scores_decay = CB * rearrange(decay, "b h c l s -> b c h l s")
    causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, device=x.device, dtype=bool), diagonal=0)
    scores_decay = scores_decay.masked_fill(~causal_mask, 0)
    out = torch.einsum('bchls,bhcs,bcshp->bclhp', scores_decay.to(x.dtype), dt.to(x.dtype),
                       rearrange(x, "b (c s) h p -> b c s h p", c=nchunks))
    state_decay_out = torch.exp(rearrange(dA_cumsum, "b h c l -> b c l h 1"))
    out_prev = torch.einsum('bclhn,bchpn->bclhp', rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
                            prev_states.to(C.dtype)) * state_decay_out
    out = out + out_prev
    out = rearrange(out, "b c l h p -> b (c l) h p")
    if D is not None:
        if D.dim() == 1:
            D = rearrange(D, "h -> h 1")
        out = out + x * D
    return out if z is None else out * F.silu(z)

def ssd_chunk_scan_combined_ref(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, dt_softplus=False, **kwargs):
  """
  Argument:
      x: (batch, seqlen, nheads, headdim)
      dt: (batch, seqlen, nheads)
      A: (nheads)
      B: (batch, seqlen, ngroups, dstate)
      C: (batch, seqlen, ngroups, dstate)
      D: (nheads, headdim) or (nheads,)
      z: (batch, seqlen, nheads, headdim)
      dt_bias: (nheads,)
  Return:
      out: (batch, seqlen, nheads, headdim)
  """
  batch, seqlen, nheads, headdim = x.shape
  dstate = B.shape[-1]
  if seqlen % chunk_size != 0:
    dt = F.pad(dt, (0, 0, 0, chunk_size - seqlen % chunk_size))
  dt = rearrange(dt, "b (c l) h -> b h c l", l=chunk_size)
  dt = dt.float()  # We want high precision for this before cumsum
  if dt_bias is not None:
    dt = dt + rearrange(dt_bias, "h -> h 1 1")
  if dt_softplus:
    dt = F.softplus(dt)
  dA = dt * rearrange(A, "h -> h 1 1")
  dA_cumsum = torch.cumsum(dA, dim=-1)
  # 1. Compute the state for each chunk
  states = chunk_state_ref(B, x, dt, dA_cumsum)
  states_dtype = states.dtype
  if states.dtype not in [torch.float32, torch.float64]:
    states = states.to(torch.float32)
  # 2. Pass the state to all the chunks by weighted cumsum.
  # state_passing_ref is much less numerically stable
  states = rearrange(state_passing_ref(rearrange(states, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1])[0],
                     "... (p n) -> ... p n", n=dstate)
  states = states.to(states_dtype)
  # 3. Compute the output for each chunk
  out = chunk_scan_ref(B, C, x, dt, dA_cumsum, states, D=D, z=z)
  return out

def mamba_split_conv1d_scan_ref(zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size,
                                dt_limit=(0.0, float("inf")), activation="silu", rmsnorm_weight=None,
                                rmsnorm_eps=1e-6, outproj_weight=None, outproj_bias=None, headdim=None, ngroups=1,
                                norm_before_gate=True):
    """
    Argument:
        zxbcdt: (batch, seqlen, 2 * dim + 2 * ngroups * dstate + nheads) where dim == nheads * headdim
        conv1d_weight: (dim + 2 * ngroups * dstate, width)
        conv1d_bias: (dim + 2 * ngroups * dstate,)
        dt_bias: (nheads,)
        A: (nheads)
        D: (nheads, headdim) or (nheads,)
        rmsnorm_weight: (dim,)
        outproj_weight: (out_dim, dim)
        outproj_bias: (out_dim,)
        headdim: if D is 1D, headdim must be passed in
        norm_before_gate: if True, we do RMSNorm(x) * F.silu(z). If False, we do RMSNorm(x * F.silu(z))
    Return:
        out: (batch, seqlen, dim)
    """
    if D.dim() == 1:
      assert headdim is not None
      nheads, = D.shape
    else:
      nheads, headdim = D.shape
    assert nheads % ngroups == 0
    batch, seqlen, _ = zxbcdt.shape
    dim = nheads * headdim
    dstate = (zxbcdt.shape[-1] - 2 * dim - nheads) // ngroups // 2
    assert zxbcdt.shape == (batch, seqlen, 2 * dim + 2 * ngroups * dstate + nheads)
    assert dt_bias.shape == (nheads,)
    assert A.shape == (nheads,)
    if rmsnorm_weight is not None:
      assert rmsnorm_weight.shape == (dim,)
    z, xBC, dt = torch.split(zxbcdt, [dim, dim + 2 * ngroups * dstate, nheads], dim=-1)
    xBC = rearrange(
      causal_conv1d_fn(rearrange(xBC, "b s d -> b d s"), conv1d_weight, conv1d_bias, activation=activation),
      "b d s -> b s d")
    x, B, C = torch.split(xBC, [dim, ngroups * dstate, ngroups * dstate], dim=-1)
    x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
    B = rearrange(B, "b l (g n) -> b l g n", g=ngroups)
    C = rearrange(C, "b l (g n) -> b l g n", g=ngroups)
    z = rearrange(z, "b l (h p) -> b l h p", h=nheads)
    out = ssd_selective_scan(x, dt.to(x.dtype), A, B, C, D=D.float(),
                             z=z if rmsnorm_weight is None else None, dt_bias=dt_bias, dt_softplus=True,
                             dt_limit=dt_limit)
    out = rearrange(out, "b s h p -> b s (h p)")
    if rmsnorm_weight is not None:
      out = rms_norm_ref(out, rmsnorm_weight, None, z=rearrange(z, "b l h p -> b l (h p)"), eps=rmsnorm_eps,
                       norm_before_gate=norm_before_gate)
    if outproj_weight is not None:
      out = F.linear(out, outproj_weight, outproj_bias)
    return out


def ssd_selective_scan(x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False, dt_limit=(0.0, float("inf"))):
  """
  Argument:
      x: (batch, seqlen, nheads, headdim)
      dt: (batch, seqlen, nheads) or (batch, seqlen, nheads, headdim)
      A: (nheads) or (dim, dstate)
      B: (batch, seqlen, ngroups, dstate)
      C: (batch, seqlen, ngroups, dstate)
      D: (nheads, headdim) or (nheads,)
      z: (batch, seqlen, nheads, headdim)
      dt_bias: (nheads,) or (nheads, headdim)
  Return:
      out: (batch, seqlen, nheads, headdim)
  """

  batch, seqlen, nheads, headdim = x.shape
  _, _, ngroups, dstate = B.shape
  x = rearrange(x, "b l h p -> b (h p) l")
  if dt.dim() == 3:
    dt = repeat(dt, "b l h -> b l h p", p=headdim)
  dt = rearrange(dt, "b l h p -> b (h p) l")
  if A.dim() == 1:
    A = repeat(A, "h -> (h p) n", p=headdim, n=dstate).to(dtype=torch.float32)
  else:
    A = A.to(dtype=torch.float32)
  B = rearrange(B, "b l g n -> b g n l")
  C = rearrange(C, "b l g n -> b g n l")
  if D is not None:
    if D.dim() == 2:
      D = rearrange(D, "h p -> (h p)")
    else:
      D = repeat(D, "h -> (h p)", p=headdim)
  if z is not None:
    z = rearrange(z, "b l h p -> b (h p) l")
  if dt_bias is not None:
    if dt_bias.dim() == 1:
      dt_bias = repeat(dt_bias, "h -> h p", p=headdim)
    dt_bias = rearrange(dt_bias, "h p -> (h p)")
  if dt_limit != (0.0, float("inf")):
    if dt_bias is not None:
      dt = dt + rearrange(dt_bias, "d -> d 1")
    if dt_softplus:
      dt = F.softplus(dt)
    dt = dt.clamp(min=dt_limit[0], max=dt_limit[1]).to(x.dtype)
    dt_bias = None
    dt_softplus = None
  out = selective_scan_ref(x, dt, A, B, C, D=D, z=z, delta_bias=dt_bias, delta_softplus=dt_softplus)
  return rearrange(out, "b (h p) l -> b l h p", p=headdim)

def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                       return_last_state=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
      delta = delta + delta_bias[..., None].float()
    if delta_softplus:
      delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    if A.is_complex():
      if is_variable_B:
        B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
      if is_variable_C:
        C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
      B = B.float()
      C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if not is_variable_B:
      deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
      if B.dim() == 3:
        deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
      else:
        B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
        deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    if is_variable_C and C.dim() == 4:
      C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    last_state = None
    for i in range(u.shape[2]):
      x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
      if not is_variable_C:
        y = torch.einsum('bdn,dn->bd', x, C)
      else:
        if C.dim() == 3:
          y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
        else:
          y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
      if i == u.shape[2] - 1:
        last_state = x
      if y.is_complex():
        y = y.real * 2
      ys.append(y)
    y = torch.stack(ys, dim=2)  # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
      out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)
