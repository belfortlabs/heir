"""Diagonal counts for real layer sizes, using only the compile-time key computation."""

import sys, time
import numpy as np
from lt import *

N = int(sys.argv[1]) if len(sys.argv) > 1 else 16384


def Z(*s):
  return np.zeros(s, dtype=np.float32)


def step(name, op, Lin, Lout=None):
  Lout = Lout or induced_layout(op, Lin)
  t = time.time()
  keys = fast_keys(op, Lin, Lout)
  dt = time.time() - t
  iters = int(np.prod(op.loops))
  print(
      f"{name:44s} ct {Lin.n_ct():3d}->{Lout.n_ct():3d}  K {len(keys):6d} "
      f" rot(bsgs) {bsgs_rotations(keys, N):5d}  iterations {iters:11d}  keys"
      f" in {dt:6.2f}s"
  )
  return Lout


print(f"N = {N}\n-- ResNet-20 (CIFAR)")
L = row_major((3, 32, 32), N)
L = step("stem 3->16 k3", conv2d(Z(16, 3, 3, 3), 32, 32, 1, 1), L)
L16 = step("16->16 k3 @32", conv2d(Z(16, 16, 3, 3), 32, 32, 1, 1), L)
L32 = step("16->32 k3 s2", conv2d(Z(32, 16, 3, 3), 32, 32, 2, 1), L16)
Ls = step("16->32 k1 s2 shortcut", conv2d(Z(32, 16, 1, 1), 32, 32, 2, 0), L16)
assert L32.key() == Ls.key()
L32 = step("32->32 k3 @16 gap 2", conv2d(Z(32, 32, 3, 3), 16, 16, 1, 1), L32)
L64 = step("32->64 k3 s2", conv2d(Z(64, 32, 3, 3), 16, 16, 2, 1), L32)
L64 = step("64->64 k3 @8 gap 4", conv2d(Z(64, 64, 3, 3), 8, 8, 1, 1), L64)
Lp = step("global avg pool 8x8", sum_pool(64, 8, 8, 8, 8), L64)
step("fc 64->10", flatten_fc(Z(10, 64), (64, 1, 1)), Lp)

print("-- ResNet-18 (ImageNet 224)")
L = row_major((3, 224, 224), N)
L = step("stem 3->64 k7 s2 p3", conv2d(Z(64, 3, 7, 7), 224, 224, 2, 3), L)
L = step(
    "3x3 s2 p1 window (one tap = one diagonal)",
    depthwise(Z(64, 3, 3), 112, 112, 2, 1),
    L,
)
L1 = step("64->64 k3 @56 gap 4", conv2d(Z(64, 64, 3, 3), 56, 56, 1, 1), L)
L2 = step("64->128 k3 s2", conv2d(Z(128, 64, 3, 3), 56, 56, 2, 1), L1)
L2 = step("128->128 k3 @28 gap 8", conv2d(Z(128, 128, 3, 3), 28, 28, 1, 1), L2)
L3 = step("128->256 k3 s2", conv2d(Z(256, 128, 3, 3), 28, 28, 2, 1), L2)
L3 = step("256->256 k3 @14 gap 16", conv2d(Z(256, 256, 3, 3), 14, 14, 1, 1), L3)
L4 = step("256->512 k3 s2", conv2d(Z(512, 256, 3, 3), 14, 14, 2, 1), L3)
L4 = step("512->512 k3 @7 gap 32", conv2d(Z(512, 512, 3, 3), 7, 7, 1, 1), L4)
Lp = step("global avg pool 7x7", sum_pool(512, 7, 7, 7, 7), L4)
step("fc 512->1000", flatten_fc(Z(1000, 512), (512, 1, 1)), Lp)

print("-- MobileNet-style block at 112x112 (after a stride-2 stem)")
L = row_major((3, 224, 224), N)
L = step("stem 3->32 k3 s2 p1", conv2d(Z(32, 3, 3, 3), 224, 224, 2, 1), L)
L = step("depthwise 32 k3 p1", depthwise(Z(32, 3, 3), 112, 112, 1, 1), L)
L = step("pointwise 32->64", conv2d(Z(64, 32, 1, 1), 112, 112), L)
L = step("depthwise 64 k3 s2 p1", depthwise(Z(64, 3, 3), 112, 112, 2, 1), L)
L = step("pointwise 64->128", conv2d(Z(128, 64, 1, 1), 56, 56), L)
