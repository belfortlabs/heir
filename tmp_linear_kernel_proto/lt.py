"""Prototype: one linear-transform kernel over strided (mixed-radix) layouts.

Layout: position p(idx) = offset + sum_digits stride * ((idx[axis] // div) % mod)
        ciphertext = p // N, slot = p % N. Optional period P (single ct, P | N).
Op:     out[M_out d + c_out] += coeff(d) * in[M_in d + c_in]  (reads outside the
        input are zero), d ranging over a box of loop extents.
"""

import math
import numpy as np


class Layout:

  def __init__(self, shape, digits, N, offset=0, period=None):
    self.shape, self.digits, self.N = tuple(shape), list(digits), N
    self.offset, self.period = offset, period

  def pos(self, idx):
    p = np.zeros(np.broadcast(*idx).shape, dtype=np.int64) + self.offset
    for ax, div, mod, stride in self.digits:
      v = idx[ax] // div
      if mod is not None:
        v = v % mod
      p = p + stride * v
    return p

  def all_pos(self):
    return self.pos(list(np.indices(self.shape)))

  def n_ct(self):
    return int(self.all_pos().max()) // self.N + 1

  def affine_stride(self, ax):
    """Stride if the position is affine in idx[ax], else None."""
    ds = [d for d in self.digits if d[0] == ax]
    if not ds:
      return 0
    if len(ds) == 1 and ds[0][1] == 1 and ds[0][2] is None:
      return ds[0][3]
    return None

  def injective(self):
    p = self.all_pos().ravel()
    return len(np.unique(p)) == len(p)

  def key(self):
    return (
        self.shape,
        tuple(sorted(self.digits, key=str)),
        self.offset,
        self.period,
    )


def row_major(shape, N, period=None):
  digits = [
      (a, 1, None, int(np.prod(shape[a + 1 :])))
      for a in range(len(shape))
      if shape[a] > 1
  ]
  return Layout(shape, digits, N, period=period)


class LinOp:

  def __init__(
      self, loops, in_shape, M_in, c_in, out_shape, M_out, c_out, coeff
  ):
    self.loops = tuple(loops)
    self.in_shape, self.out_shape = tuple(in_shape), tuple(out_shape)
    self.M_in, self.c_in = np.array(M_in, dtype=np.int64), np.array(
        c_in, dtype=np.int64
    )
    self.M_out, self.c_out = np.array(M_out, dtype=np.int64), np.array(
        c_out, dtype=np.int64
    )
    self.coeff = coeff

  def points(self, d):
    ii = [
        sum(int(m) * d[j] for j, m in enumerate(row) if m) + int(c)
        for row, c in zip(self.M_in, self.c_in)
    ]
    oi = [
        sum(int(m) * d[j] for j, m in enumerate(row) if m) + int(c)
        for row, c in zip(self.M_out, self.c_out)
    ]
    ii = [np.zeros(d[0].shape, dtype=np.int64) + v for v in ii]
    oi = [np.zeros(d[0].shape, dtype=np.int64) + v for v in oi]
    valid = np.ones(d[0].shape, dtype=bool)
    for v, n in zip(ii, self.in_shape):
      valid &= (v >= 0) & (v < n)
    return ii, oi, valid

  def domain(self):
    return [g.ravel() for g in np.indices(self.loops)]

  def naive(self, x):
    d = self.domain()
    ii, oi, valid = self.points(d)
    y = np.zeros(self.out_shape)
    np.add.at(
        y,
        tuple(v[valid] for v in oi),
        self.coeff(d)[valid] * x[tuple(v[valid] for v in ii)],
    )
    return y


# ---------------------------------------------------------------- kernel


def modulus(Lin, Lout):
  """(Q, write_follows_input): shifts are taken mod Q."""
  N = Lout.N
  if Lout.period and (not Lin.period or Lout.period <= Lin.period):
    return Lout.period, True
  if Lin.period:
    return Lin.period, False
  return N, False


def pack(op, Lin, Lout):
  """What the rolled runtime packing loop computes: keys and K x N diagonals."""
  N = Lout.N
  d = op.domain()
  ii, oi, valid = op.points(d)
  pi = Lin.pos(ii)[valid]
  po = Lout.pos(oi)[valid]
  w = op.coeff(d)[valid]
  Q, follow = modulus(Lin, Lout)
  r = (pi - po) % Q
  slot = (pi - r) % N if follow else po % N
  bo = np.zeros_like(po) if follow else po // N
  keys, inv = np.unique(np.stack([bo, pi // N, r]), axis=1, return_inverse=True)
  diags = np.zeros((keys.shape[1], N))
  np.add.at(diags, (inv.ravel(), slot), w)
  return [tuple(int(v) for v in k) for k in keys.T], diags


def evaluate(keys, diags, X, Lout):
  N = Lout.N
  Y = np.zeros((Lout.n_ct(), N))
  for k, (bo, bi, r) in enumerate(keys):
    Y[bo] += diags[k] * np.roll(X[bi], -r)
  if Lout.period:
    Y = sum(
        np.roll(Y, m * Lout.period, axis=1) for m in range(N // Lout.period)
    )
  return Y


def encode(x, L, rng):
  """Unused slots hold garbage, to check they are never read."""
  X = rng.normal(size=(L.n_ct(), L.N)) * 100
  p = L.all_pos().ravel()
  if L.period:
    for m in range(L.N // L.period):
      X.reshape(-1)[(p + m * L.period) % L.N] = x.ravel()
  else:
    X.reshape(-1)[p] = x.ravel()
  return X


def decode(Y, L):
  return Y.reshape(-1)[L.all_pos().ravel()].reshape(L.shape)


# ------------------------------------------- compile-time key computation


def aligned_dims(op, Lin, Lout):
  """Loop dims on which the shift p_in - p_out does not depend, with the
  coefficient of the output position on each."""
  res = {}
  for j in range(len(op.loops)):
    ok, cin, cout = True, 0, 0
    for i in np.nonzero(op.M_in[:, j])[0]:
      s = Lin.affine_stride(i)
      ok &= s is not None
      cin += (s or 0) * int(op.M_in[i, j])
    for i in np.nonzero(op.M_out[:, j])[0]:
      s = Lout.affine_stride(i)
      ok &= s is not None
      cout += (s or 0) * int(op.M_out[i, j])
    if ok and cin == cout and cout > 0:
      res[j] = cout
  # the bounds of each input axis must give an interval per aligned dim
  for i in range(len(op.in_shape)):
    js = [j for j in res if op.M_in[i, j]]
    for j in js[1:]:
      del res[j]
    if js and op.M_in[i, js[0]] < 0:
      del res[js[0]]
  return res


def fast_keys(op, Lin, Lout):
  """Key set without visiting the aligned dims' points one by one."""
  N = Lout.N
  Q, follow = modulus(Lin, Lout)
  al = aligned_dims(op, Lin, Lout)
  if Q != N or follow:
    al = {}
  A = sorted(al, key=lambda j: -al[j])
  U = [j for j in range(len(op.loops)) if j not in al]
  grids = (
      np.indices([op.loops[j] for j in U]).reshape(len(U), -1)
      if U
      else np.zeros((0, 1), dtype=np.int64)
  )
  nU = grids.shape[1]
  d0 = [np.zeros(nU, dtype=np.int64) for _ in op.loops]
  for k, j in enumerate(U):
    d0[j] = grids[k]
  ii0, oi0, _ = op.points(d0)
  pi0, po0 = Lin.pos(ii0), Lout.pos(oi0)
  delta = pi0 - po0
  valid = np.ones(nU, dtype=bool)
  lo = {j: np.zeros(nU, dtype=np.int64) for j in A}
  hi = {j: np.full(nU, op.loops[j] - 1, dtype=np.int64) for j in A}
  for i, n in enumerate(op.in_shape):
    js = [j for j in A if op.M_in[i, j]]
    if not js:
      valid &= (ii0[i] >= 0) & (ii0[i] < n)
      continue
    j, m = js[0], int(op.M_in[i, js[0]])
    lo[j] = np.maximum(lo[j], -((ii0[i]) // m))  # ceil(-ii0/m)
    hi[j] = np.minimum(hi[j], (n - 1 - ii0[i]) // m)
  out = []

  def emit(mask, bo, bi):
    if mask.any():
      out.append(np.stack([bo[mask], bi[mask], (delta % Q)[mask]]))

  if Q != N or follow:  # periodic: single output ct
    emit(valid, np.zeros(nU, dtype=np.int64), pi0 // N)
  elif not A:
    emit(valid, po0 // N, pi0 // N)
  else:
    inner, outer = A[-1], A[:-1]
    c = al[inner]
    for combo in np.ndindex(*[op.loops[j] for j in outer]) if outer else [()]:
      mask = valid.copy()
      a = po0 + c * lo[inner]
      for j, v in zip(outer, combo):
        mask &= (lo[j] <= v) & (v <= hi[j])
        a = a + al[j] * v
      n = hi[inner] - lo[inner] + 1
      mask &= n > 0
      if not mask.any():
        continue
      last = a + (n - 1) * c
      emit(mask, a // N, (a + delta) // N)
      for base in (a, a + delta):  # crossings of multiples of N
        ncross = int(((last - a)[mask] // N).max()) + 1
        for t in range(1, ncross + 1):
          m = base // N + t
          i = -((base - m * N) // c)  # ceil((mN - base)/c)
          mk = mask & (i <= n - 1)
          p = a + i * c
          emit(mk, p // N, (p + delta) // N)
  k = np.unique(np.concatenate(out, axis=1), axis=1)
  return [tuple(int(v) for v in x) for x in k.T]


# ------------------------------------------------- induced output layout


def induced_layout(
    op, Lin, period_for_free=True, compact=True, _identical=True
):
  """Output layout that keeps paired dims on the input's grid and places the
  remaining output dims in the free positions (gaps first, then planes)."""
  N = Lin.N
  digits, occupied, free_axes, paired_in = [], [], [], set()
  for j, n in enumerate(op.out_shape):
    if n == 1:
      continue
    vs = np.nonzero(op.M_out[j])[0]
    done = False
    if len(vs) == 1 and op.M_out[j, vs[0]] == 1 and op.c_out[j] == 0:
      v = vs[0]
      axes = np.nonzero(op.M_in[:, v])[0]
      if len(axes) == 1:
        i, s = axes[0], int(op.M_in[axes[0], v])
        sig = Lin.affine_stride(i)
        others = [k for k in np.nonzero(op.M_in[i])[0] if k != v]
        if sig and s > 0:
          digits.append((j, 1, None, s * sig))
          occupied.append((s * sig, n))
          paired_in.add(i)
          done = True
        elif _identical and s == 1 and not others and op.c_in[i] == 0:
          for ax, div, mod, st in Lin.digits:
            if ax == i:
              digits.append((j, div, mod, st))
              occupied.append((st, mod if mod else -(-n // div)))
          paired_in.add(i)
          done = True
    if not done:
      free_axes.append(j)
  occupied.sort()
  free, cursor = [], 1
  for st, ext in occupied:
    if st >= 2 * cursor:
      free.append((cursor, st // cursor))
    cursor = max(cursor, st * ext)
  cand = [
      st for ax, _, _, st in Lin.digits if ax not in paired_in and st >= cursor
  ]
  top = min(cand) if cand else cursor
  for j in reversed(free_axes):
    n, div = op.out_shape[j], 1
    while free and div < n:
      st, rho = free.pop(0)
      digits.append((j, div, rho, st))
      div *= rho
    if div < n:
      digits.append((j, div, None, top))
      top *= -(-n // div)
  L = Layout(op.out_shape, digits, N, offset=Lin.offset)
  total = int(np.prod(op.out_shape))
  if period_for_free and not occupied and total <= N:
    L = row_major(
        op.out_shape, N, period=1 << max(0, math.ceil(math.log2(total)))
    )
  if not L.injective():
    L = row_major(op.out_shape, N)
  if compact and _identical and total < (int(L.all_pos().max()) + 1) // 2:
    # keeping channels in place would leave the result sparse: let them fill the gaps
    L2 = induced_layout(op, Lin, period_for_free, compact, _identical=False)
    if L2.n_ct() < L.n_ct() or L.n_ct() == 1:
      return L2
  return L


# --------------------------------------------------------------- op zoo


def conv2d(w, H, W, stride=1, pad=0, dil=1, groups=1):
  """w: (Co, Ci/groups, kh, kw); tensors are (C, H, W)."""
  Co, cig, kh, kw = w.shape
  cog = Co // groups
  Ho = (H + 2 * pad - dil * (kh - 1) - 1) // stride + 1
  Wo = (W + 2 * pad - dil * (kw - 1) - 1) // stride + 1
  # loops: g, cog, cig, kh, kw, ho, wo
  loops = (groups, cog, cig, kh, kw, Ho, Wo)
  M_in = [
      [cig, 0, 1, 0, 0, 0, 0],
      [0, 0, 0, dil, 0, stride, 0],
      [0, 0, 0, 0, dil, 0, stride],
  ]
  M_out = [
      [cog, 1, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 1, 0],
      [0, 0, 0, 0, 0, 0, 1],
  ]
  return LinOp(
      loops,
      (cig * groups, H, W),
      M_in,
      [0, -pad, -pad],
      (Co, Ho, Wo),
      M_out,
      [0, 0, 0],
      lambda d: w[d[0] * cog + d[1], d[2], d[3], d[4]],
  )


def depthwise(w, H, W, stride=1, pad=0):
  """w: (C, kh, kw); the channel is one loop, as in depthwise_conv_2d_nchw_chw."""
  C, kh, kw = w.shape
  Ho = (H + 2 * pad - kh) // stride + 1
  Wo = (W + 2 * pad - kw) // stride + 1
  M_in = [[1, 0, 0, 0, 0], [0, 1, 0, stride, 0], [0, 0, 1, 0, stride]]
  M_out = [[1, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
  return LinOp(
      (C, kh, kw, Ho, Wo),
      (C, H, W),
      M_in,
      [0, -pad, -pad],
      (C, Ho, Wo),
      M_out,
      [0, 0, 0],
      lambda d: w[d[0], d[1], d[2]],
  )


def sum_pool(C, H, W, k, stride, scale=1.0):
  return depthwise(np.full((C, k, k), scale), H, W, stride)


def matvec(w):
  R, C = w.shape
  return LinOp(
      (R, C), (C,), [[0, 1]], [0], (R,), [[1, 0]], [0], lambda d: w[d[0], d[1]]
  )


def flatten_fc(w, shape):
  """y = w @ x.reshape(-1), reading x through its own (C,H,W) layout."""
  R = w.shape[0]
  C, H, W = shape
  w4 = w.reshape(R, C, H, W)
  M_in = [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
  return LinOp(
      (R, C, H, W),
      shape,
      M_in,
      [0, 0, 0],
      (R,),
      [[1, 0, 0, 0]],
      [0],
      lambda d: w4[d[0], d[1], d[2], d[3]],
  )


def broadcast(C, H, W):
  """out[c,h,w] = in[c,0,0]."""
  return LinOp(
      (C, H, W),
      (C, 1, 1),
      [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
      [0, 0, 0],
      (C, H, W),
      np.eye(3, dtype=np.int64),
      [0, 0, 0],
      lambda d: np.ones(d[0].shape),
  )


def run(op, Lin, Lout, rng):
  """Returns (max abs error, K, keys agree)."""
  x = rng.normal(size=op.in_shape)
  keys, diags = pack(op, Lin, Lout)
  Y = evaluate(keys, diags, encode(x, Lin, rng), Lout)
  err = np.abs(decode(Y, Lout) - op.naive(x)).max()
  if Lout.period:  # every copy must hold the result
    p = Lout.all_pos().ravel()
    for m in range(Lout.N // Lout.period):
      err = max(
          err,
          np.abs(
              Y.reshape(-1)[(p + m * Lout.period) % Lout.N]
              - op.naive(x).ravel()
          ).max(),
      )
  fk = fast_keys(op, Lin, Lout)
  return err, len(keys), set(fk) == set(keys), len(fk)


def bsgs_rotations(keys, N):
  """Fewest rotations over power-of-two giant steps (baby steps shared per input ct)."""
  best = None
  g = 1
  while g <= N:
    baby = {(bi, r % g) for _, bi, r in keys if r % g}
    giant = {(bo, r - r % g) for bo, _, r in keys if r - r % g}
    n = len(baby) + len(giant)
    best = n if best is None else min(best, n)
    g *= 2
  return best
