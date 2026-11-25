"""Patches a check driver (gen_driver.py output) so that it also writes the
secret input, the reference output and the plaintext-backend output to text
files in the current directory.

usage: ref_dump.py driver.cpp > driver_dump.cpp
"""

import re
import sys

src = open(sys.argv[1]).read()
secret = re.search(
    r'for \(auto& v : (\w+)_master\) v = secretDist\(rng\);', src
).group(1)
out_size = re.search(r'for \(int64_t i = 0; i < (\d+); \+\+i\)', src).group(1)
dump = f"""
  {{
    FILE* f = std::fopen("input.txt", "w");
    for (auto v : {secret}_master) std::fprintf(f, "%.9g\\n", (double)v);
    std::fclose(f);
    f = std::fopen("expected.txt", "w");
    for (int64_t i = 0; i < {out_size}; ++i)
      std::fprintf(f, "%.9g\\n", (double)expected.data[expected.offset + i]);
    std::fclose(f);
    f = std::fopen("plain.txt", "w");
    for (int64_t i = 0; i < {out_size}; ++i)
      std::fprintf(f, "%.9g\\n", (double)actual.data[actual.offset + i]);
    std::fclose(f);
  }}
"""
marker = '  double maxError = 0, maxValue = 0;'
assert marker in src
print(src.replace(marker, dump + marker))
