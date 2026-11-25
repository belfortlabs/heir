import sys, glob, os, re, collections

d = sys.argv[1]
print(
    "model\ttime\tstatus\tboot\tLTs\tLT_diags\tdistinct_diags\trot\tmul_plain\tct_mul\trescale\tcheb"
)
for f in sorted(glob.glob(d + "/*.summary")):
  m = os.path.basename(f)[:-8]
  lines = open(f).read().splitlines()
  t = re.search(r"time=(\d+)s", lines[0]).group(1)
  rc = re.search(r"rc=(\d+)", lines[0]).group(1)
  err = lines[1][:70] if len(lines) > 1 else ""
  c = collections.defaultdict(lambda: collections.defaultdict(int))
  for l in open(d + "/" + m + ".counts", errors="replace"):
    p = l.rstrip("\n").split("\t")
    if len(p) == 4 and p[0] == "count-ops":
      c[p[1]][p[2]] = int(p[3])
  main = max(
      c,
      key=lambda k: sum(
          v
          for n, v in c[k].items()
          if n.startswith("ckks.") or n.startswith("kernel.")
      ),
      default=None,
  )
  if main is None or rc != "0":
    print(f"{m}\t{t}\tFAIL rc={rc} {err}")
    continue
  x = c[main]
  cheb = " ".join(
      n.split("degree ")[1].rstrip(")") + ":" + str(v)
      for n, v in sorted(x.items())
      if "degree" in n
  )
  print(
      "\t".join(
          str(v)
          for v in [
              m,
              t,
              "ok",
              x["ckks.bootstrap"],
              x["kernel.apply_linear_transform"],
              x["kernel.apply_linear_transform(diagonals)"],
              x["distinct-transform-diagonals"],
              x["ckks.rotate"],
              x["ckks.mul_plain"],
              x["ckks.mul"],
              x["ckks.rescale"],
              cheb,
          ]
      )
  )
