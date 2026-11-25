import numpy as np
from lt import *

rng = np.random.default_rng(0)
rows = []


def check(name, op, Lin, Lout=None):
  Lout = Lout or induced_layout(op, Lin)
  assert Lout.injective(), name
  err, K, agree, fk = run(op, Lin, Lout, rng)
  rows.append((name, Lin.n_ct(), Lout.n_ct(), K, fk, agree, err))
  assert err < 1e-9, (name, err)
  return Lout


def W(*s):
  return rng.normal(size=s)


N = 512
# CIFAR-like chain at 8x8, 1 ciphertext per tensor
L0 = row_major((3, 8, 8), N)
L1 = check("conv 3->8 k3 s1 p1", conv2d(W(8, 3, 3, 3), 8, 8, 1, 1), L0)
L2 = check(
    "conv 8->16 k3 s2 p1 (gap 1->2)", conv2d(W(16, 8, 3, 3), 8, 8, 2, 1), L1
)
L2s = check(
    "conv 8->16 k1 s2 (shortcut)", conv2d(W(16, 8, 1, 1), 8, 8, 2, 0), L1
)
assert L2.key() == L2s.key(), "shortcut and main path layouts differ"
L3 = check(
    "conv 16->16 k3 s1 p1 on gap 2", conv2d(W(16, 16, 3, 3), 4, 4, 1, 1), L2
)
assert L3.key() == L2.key()
L4 = check(
    "depthwise k3 s2 p1 on gap 2", depthwise(W(16, 3, 3), 4, 4, 2, 1), L3
)
L5 = check(
    "conv 16->64 k1 after depthwise s2", conv2d(W(64, 16, 1, 1), 2, 2), L4
)
check(
    "grouped conv 16->16 g4 k3 p1",
    conv2d(W(16, 4, 3, 3), 4, 4, 1, 1, groups=4),
    L3,
)
check(
    "dilated conv 8->8 k3 d2 p2", conv2d(W(8, 8, 3, 3), 8, 8, 1, 2, dil=2), L1
)
L6 = check("avg pool 2x2 s2", sum_pool(8, 8, 8, 2, 2, 0.25), L1)
check(
    "conv 8->8 k3 p1 after pool (gap 2, sparse)",
    conv2d(W(8, 8, 3, 3), 4, 4, 1, 1),
    L6,
)
Lg = check("global pool on gap 2", sum_pool(16, 4, 4, 4, 4, 1 / 16), L3)
Lf = check(
    "fc 16->10 from pooled (periodic out)",
    flatten_fc(W(10, 16), (16, 1, 1)),
    Lg,
)
check("fc 10->7 periodic in and out", matvec(W(7, 10)), Lf)
check("fc 10->40 periodic in, wider out", matvec(W(40, 10)), Lf)
check(
    "fc 256->10 from gap-2 feature map", flatten_fc(W(10, 256), (16, 4, 4)), L3
)
check("broadcast C,1,1 -> C,4,4 on gap 2", broadcast(16, 4, 4), Lg)

# odd sizes, valid convs
Lo = row_major((4, 7, 7), N)
check("conv 4->8 k3 s2 p1 on 7x7 (odd)", conv2d(W(8, 4, 3, 3), 7, 7, 2, 1), Lo)
Lv = row_major((2, 12, 12), N)
Lv1 = check("valid conv 2->3 k5 on 12x12", conv2d(W(3, 2, 5, 5), 12, 12), Lv)
check(
    "valid conv 3->3 k3 on 8x8 (stays on 12-grid)",
    conv2d(W(3, 3, 3, 3), 8, 8),
    Lv1,
)

# planes larger than a ciphertext: channels straddle ciphertexts
N = 128
Ls = row_major((3, 12, 12), N)
Ls1 = check(
    "multi-ct conv 3->6 k3 p1, plane 144 > N 128",
    conv2d(W(6, 3, 3, 3), 12, 12, 1, 1),
    Ls,
)
Ls2 = check(
    "multi-ct conv 6->12 k3 s2 p1", conv2d(W(12, 6, 3, 3), 12, 12, 2, 1), Ls1
)
check(
    "multi-ct depthwise k3 p1 on gap 2", depthwise(W(12, 3, 3), 6, 6, 1, 1), Ls2
)
check("multi-ct fc 432->5", flatten_fc(W(5, 432), (12, 6, 6)), Ls2)

# arbitrary (non-induced) layout pairs: correct, only K changes
N = 512
check(
    "conv 3->8 into row-major output (misaligned planes ok)",
    conv2d(W(8, 3, 3, 3), 8, 8, 1, 1),
    L0,
    row_major((8, 8, 8), N),
)
check(
    "conv s2 into row-major output (misaligned)",
    conv2d(W(16, 8, 3, 3), 8, 8, 2, 1),
    L1,
    row_major((16, 4, 4), N),
)
check(
    "gap-2 -> row-major identity (layout conversion)",
    LinOp(
        (16, 4, 4),
        (16, 4, 4),
        np.eye(3),
        [0, 0, 0],
        (16, 4, 4),
        np.eye(3),
        [0, 0, 0],
        lambda d: np.ones(d[0].shape),
    ),
    L2,
    row_major((16, 4, 4), N),
)

print(f"{'op':52s} ct_in ct_out    K  K_fast keys_ok  max_err")
for r in rows:
  print(
      f"{r[0]:52s} {r[1]:5d} {r[2]:6d} {r[3]:4d} {r[4]:7d} {str(r[5]):7s} {r[6]:.1e}"
  )
assert all(r[5] for r in rows), "fast key set differs from enumeration"
print("all ok")
