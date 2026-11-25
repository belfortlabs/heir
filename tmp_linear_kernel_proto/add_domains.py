"""Writes a copy of a corpus model whose compare-and-select activations carry
an approximation domain.

The corpus models have no `domain_lower`/`domain_upper` attributes, so every
ReLU, ReLU6, hard-sigmoid and hard-swish is approximated on [-1, 1]. A hard
sigmoid clamps `x + 3`, which is far outside of that, and the approximating
polynomial overflows in the reference as well. With a domain the reference
is finite and the numeric check has a verdict.

usage: add_domains.py model.mlir bound > model_with_domains.mlir
"""

import sys

lines = open(sys.argv[1]).read().split('\n')
bound = float(sys.argv[2])
attrs = f' attrs =  {{domain_lower = {-bound}, domain_upper = {bound}}} {{'
for i, line in enumerate(lines):
  if 'linalg.generic' not in line or not line.rstrip().endswith('{'):
    continue
  for body in lines[i + 1 : i + 40]:
    if 'linalg.yield' in body:
      break
    if 'arith.cmpf' in body:
      lines[i] = line.rstrip()[:-2] + attrs
      break
print('\n'.join(lines))
