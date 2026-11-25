import sys, glob, os, re, collections

d = sys.argv[1]
print(
    "model\ttime\tstatus\tboot\trot\tshifts\tdynrot\tmul_plain\tct_mul\trelin\trescale\tadd\tcheb(deg:count)"
)
for f in sorted(glob.glob(d + "/*.summary")):
  m = os.path.basename(f)[:-8]
  lines = open(f).read().splitlines()
  t = re.search(r"time=(\d+)s", lines[0]).group(1)
  rc = re.search(r"rc=(\d+)", lines[0]).group(1)
  err = lines[1][:90] if len(lines) > 1 else ""
  c = collections.defaultdict(lambda: collections.defaultdict(int))
  for l in open(d + "/" + m + ".counts", errors="replace"):
    p = l.rstrip("\n").split("\t")
    if len(p) == 4 and p[0] == "count-ops":
      c[p[1]][p[2]] = int(p[3])
  # the compiled function: most ckks ops, not a helper
  main = max(
      c,
      key=lambda k: sum(v for n, v in c[k].items() if n.startswith("ckks.")),
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
              x["ckks.rotate"],
              x["distinct-shifts"],
              x["ckks.rotate(dynamic shift)"],
              x["ckks.mul_plain"],
              x["ckks.mul"],
              x["ckks.relinearize"],
              x["ckks.rescale"],
              x["ckks.add"]
              + x["ckks.sub"]
              + x["ckks.add_plain"]
              + x["ckks.sub_plain"],
              cheb,
          ]
      )
  )
