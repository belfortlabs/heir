"""End to end: a small residual network evaluated only through the kernel, against numpy.
Planes (144 positions) are larger than a ciphertext (128 slots)."""

import numpy as np
from lt import *

rng = np.random.default_rng(1)
N = 128
total_K = 0


def lin(op, X, Lin, Lout=None):
  global total_K
  Lout = Lout or induced_layout(op, Lin)
  keys, diags = pack(op, Lin, Lout)
  assert set(keys) == set(fast_keys(op, Lin, Lout))
  total_K += len(keys)
  return evaluate(keys, diags, X, Lout), Lout


def W(*s):
  return rng.normal(size=s) * 0.3


act = lambda v: np.tanh(v) + 0.1 * v  # slotwise, also applied to unused slots

w0, w1, w2, w3, ws, wd, wp, wf = (
    W(6, 3, 3, 3),
    W(6, 6, 3, 3),
    W(12, 6, 3, 3),
    W(12, 12, 3, 3),
    W(12, 6, 1, 1),
    W(12, 3, 3),
    W(24, 12, 1, 1),
    W(5, 24),
)
x = rng.normal(size=(3, 12, 12))

# reference
r = act(conv2d(w0, 12, 12, 1, 1).naive(x))
r = act(conv2d(w1, 12, 12, 1, 1).naive(r) + r)  # identity residual
m = conv2d(w3, 6, 6, 1, 1).naive(act(conv2d(w2, 12, 12, 2, 1).naive(r)))
r = act(m + conv2d(ws, 12, 12, 2, 0).naive(r))  # strided residual
r = act(
    conv2d(wp, 6, 6).naive(depthwise(wd, 6, 6, 1, 1).naive(r))
)  # separable conv
r = sum_pool(24, 6, 6, 6, 6, 1 / 36).naive(r)
ref = matvec(wf).naive(r.reshape(-1))

# through the kernel
L0 = row_major((3, 12, 12), N)
X = encode(x, L0, rng)
A, La = lin(conv2d(w0, 12, 12, 1, 1), X, L0)
A = act(A)
B, Lb = lin(conv2d(w1, 12, 12, 1, 1), A, La)
assert Lb.key() == La.key()
A = act(B + A)
M, Lm = lin(conv2d(w2, 12, 12, 2, 1), A, La)
M, Lm2 = lin(conv2d(w3, 6, 6, 1, 1), act(M), Lm)
S, Ls = lin(conv2d(ws, 12, 12, 2, 0), A, La)
assert Lm2.key() == Ls.key(), "residual operands must share a layout"
A = act(M + S)
D, Ld = lin(depthwise(wd, 6, 6, 1, 1), A, Ls)
P, Lp = lin(conv2d(wp, 6, 6), D, Ld)
G, Lg = lin(sum_pool(24, 6, 6, 6, 6, 1 / 36), act(P), Lp)
F, Lf = lin(flatten_fc(wf, (24, 1, 1)), G, Lg)
out = decode(F, Lf)
print("ciphertexts per tensor:", [l.n_ct() for l in (L0, La, Lm, Lp, Lg, Lf)])
print(
    "total diagonals:",
    total_K,
    " max abs error:",
    np.abs(out - ref).max(),
    " |ref| max:",
    np.abs(ref).max(),
)
assert np.abs(out - ref).max() < 1e-8
print("ok")
