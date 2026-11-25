"""Per model: run every conv/pool of a corpus file through the compile-time key
computation with multiplexed layouts on the input image's grid. Estimates only:
layouts are built from shapes (gap = largest power of two with H*gap <= grid),
not propagated through the dataflow graph."""

import glob, math, os, re, sys, time
from collections import Counter
import numpy as np
from lt import Layout, LinOp, fast_keys, bsgs_rotations

corpus = os.path.expanduser(
    '~/Code/heir-worktrees/isl-session-archive/tmp_isl/corpus/'
)
N = int(sys.argv[2]) if len(sys.argv) > 2 else 16384


def p2floor(x):
  return 1 << max(0, int(math.floor(math.log2(max(1, x)))))


def mux(C, H, W, gy, gx, Hp, Wp):
  digits = []
  if H > 1:
    digits.append((1, 1, None, gy * Wp))
  if W > 1:
    digits.append((2, 1, None, gx))
  div = 1
  if gx > 1:
    digits.append((0, div, gx, 1))
    div *= gx
  if gy > 1:
    digits.append((0, div, gy, Wp))
    div *= gy
  if C > div:
    digits.append((0, div, None, Hp * Wp))
  return Layout((C, H, W), digits, N)


def conv_op(Ci, Co, groups, kh, kw, sy, sx, dy, dx, py, px, H, W, Ho, Wo):
  cig, cog = Ci // groups, Co // groups
  loops = (groups, cog, cig, kh, kw, Ho, Wo)
  M_in = [
      [cig, 0, 1, 0, 0, 0, 0],
      [0, 0, 0, dy, 0, sy, 0],
      [0, 0, 0, 0, dx, 0, sx],
  ]
  M_out = [
      [cog, 1, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 1, 0],
      [0, 0, 0, 0, 0, 0, 1],
  ]
  return LinOp(
      loops,
      (Ci, H, W),
      M_in,
      [0, -py, -px],
      (Co, Ho, Wo),
      M_out,
      [0, 0, 0],
      None,
  )


def shape(s):
  return [int(v) for v in s.split('x')[:-1]]


def dense(s):
  v = [int(t) for t in s.strip('[]').split(',')]
  return v * 2 if len(v) == 1 else v


def run(f):
  name = os.path.basename(f)[8:-5]
  H0 = W0 = None
  tot_k = tot_rot = nops = maxct = 0
  tkeys = 0.0
  other = Counter()
  worst = (0, '')
  with open(f, errors='replace') as fh:
    for line in fh:
      if len(line) > 4000:
        line = line[:4000]
      if H0 is None and 'func.func' in line:
        m = re.search(r'tensor<([0-9x]+x[a-z0-9]+)> \{secret', line)
        s = shape(m.group(1))
        H0, W0 = (s[-2], s[-1]) if len(s) >= 3 else (1, s[-1])
      m = re.search(r'= linalg\.(\w+)', line)
      if not m:
        continue
      kind = m.group(1)
      if kind in ('matmul', 'matvec', 'vecmat'):
        tys = re.findall(r'tensor<([0-9x]+x)f32>', line)
        r = shape(tys[-1] + 'f')[-1]
        k = 1 << math.ceil(math.log2(max(2, r)))
        tot_k += k
        tot_rot += int(2 * math.sqrt(k))
        nops += 1
        continue
      if kind not in (
          'conv_2d_nchw_fchw',
          'depthwise_conv_2d_nchw_chw',
          'conv_2d_ngchw_gfchw',
          'pooling_nchw_sum',
          'pooling_nchw_max',
      ):
        other[kind] += 1
        continue
      dil = dense(re.search(r'dilations = dense<(.*?)>', line).group(1))
      st = dense(re.search(r'strides = dense<(.*?)>', line).group(1))
      tys = re.findall(r'tensor<([0-9x]+x)f32>', line)
      i, w, o = shape(tys[0] + 'f'), shape(tys[1] + 'f'), shape(tys[-1] + 'f')
      if kind == 'conv_2d_nchw_fchw':
        Ci, Co, kh, kw, g = i[1], w[0], w[2], w[3], 1
      elif kind == 'depthwise_conv_2d_nchw_chw':
        Ci, Co, kh, kw, g = i[1], i[1], w[1], w[2], i[1]
      elif kind == 'conv_2d_ngchw_gfchw':
        Ci, Co, kh, kw, g = i[1] * i[2], w[0] * w[1], w[3], w[4], i[1]
      else:
        Ci, Co, kh, kw, g = i[1], i[1], w[0], w[1], i[1]
      Hip, Wip, Ho, Wo = i[-2], i[-1], o[-2], o[-1]
      gyo, gxo = p2floor(H0 / Ho), p2floor(W0 / Wo)
      gyi, gxi = max(1, gyo // st[0]), max(1, gxo // st[1])
      H, W = min(Hip, H0 // gyi), min(Wip, W0 // gxi)
      py, px = (Hip - H) // 2, (Wip - W) // 2
      Lin, Lout = mux(Ci, H, W, gyi, gxi, H0, W0), mux(
          Co, Ho, Wo, gyo, gxo, H0, W0
      )
      op = conv_op(
          Ci, Co, g, kh, kw, st[0], st[1], dil[0], dil[1], py, px, H, W, Ho, Wo
      )
      t = time.time()
      keys = fast_keys(op, Lin, Lout)
      tkeys += time.time() - t
      if kind == 'pooling_nchw_max':
        k, rot = kh * kw, kh * kw  # one aligned copy per tap, then a max tree
      else:
        k, rot = len(keys), bsgs_rotations(keys, N)
      cts = max(Lin.n_ct(), Lout.n_ct())
      maxct = max(maxct, cts)
      tot_k += k
      tot_rot += rot
      nops += 1
      if k > worst[0]:
        worst = (k, f'{kind} {Ci}->{Co} k{kh}x{kw} s{st[0]} @{Ho}x{Wo}')
  oth = ','.join(f'{k}:{v}' for k, v in other.most_common(6))
  print(
      f'{name}\t{nops}\t{maxct}\t{tot_k}\t{tot_rot}\t{tkeys:.1f}\t{worst[0]} ({worst[1]})\t{oth}',
      flush=True,
  )


if __name__ == '__main__':
  run(sys.argv[1])
