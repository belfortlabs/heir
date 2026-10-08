"""Checks that HEIR's copy of cyclops_planner.h matches Cyclops'.

python3 bazel/cyclops/header_check.py CYCLOPS_DIR

CYCLOPS_DIR is a Cyclops checkout or an unpacked Cyclops release. The copy must
declare Cyclops' functions with the same signatures, its constants must have
Cyclops' values, and its structs Cyclops' fields.
"""

import re
import sys
from pathlib import Path

HEADER = Path(__file__).with_name("cyclops_planner.h")


def normalize(text):
  text = re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)
  text = re.sub(r"\s+", " ", text)
  return re.sub(r" ?([*,()\[\];{}]) ?", r"\1", text)


def functions(header):
  """Function name -> normalized prototype."""
  return {
      name: f"{result.strip()} {name}({params})"
      for result, name, params in re.findall(
          r"(?:^|(?<=[;{}]))([\w *]+?)\b(cyclops_\w+)\(([^)]*)\);",
          normalize(header),
      )
  }


def constants(header):
  """Enumerator -> value, counting implicit values from the previous one."""
  values = {}
  for body in re.findall(r"enum\{([^}]*)\}", normalize(header)):
    value = -1
    for item in body.split(","):
      name, _, explicit = item.partition("=")
      value = int(explicit) if explicit else value + 1
      values[name.strip()] = value
  return values


def structs(header):
  """Struct name -> normalized fields."""
  return {
      name: fields
      for fields, name in re.findall(
          r"typedef struct\{([^}]*)\}(\w+);", normalize(header)
      )
  }


def check_cyclops(header, cyclops):
  errors = []
  ours, theirs = functions(header), functions(cyclops)
  for name in sorted(ours.keys() | theirs.keys()):
    if ours.get(name) != theirs.get(name):
      errors.append(
          f"{name}: copy {ours.get(name)} vs Cyclops {theirs.get(name)}"
      )
  their_constants = constants(cyclops)
  for name, value in constants(header).items():
    if their_constants.get(name) != value:
      errors.append(
          f"{name} = {value}, Cyclops has {their_constants.get(name)}"
      )
  their_structs = structs(cyclops)
  for name, fields in structs(header).items():
    if their_structs.get(name) != fields:
      errors.append(
          f"struct {name}: {fields} vs Cyclops {their_structs.get(name)}"
      )
  return errors


def main(argv):
  (cyclops_dir,) = argv[1:]
  candidates = [
      Path(cyclops_dir, "planner/include/cyclops_planner.h"),
      Path(cyclops_dir, "include/cyclops_planner.h"),
  ]
  cyclops_header = next((path for path in candidates if path.exists()), None)
  if cyclops_header is None:
    print(f"no cyclops_planner.h in {cyclops_dir}")
    return 1
  header = HEADER.read_text()
  errors = check_cyclops(header, cyclops_header.read_text())
  if not functions(header):
    errors.append("no functions found in the header")
  for error in errors:
    print(error)
  return 1 if errors else 0


if __name__ == "__main__":
  sys.exit(main(sys.argv))
