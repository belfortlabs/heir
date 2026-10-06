"""Checks that HEIR's copy of cyclops_planner.h matches the stub and Cyclops.

header_check.py HEADER --stub planner_stub.cc
  every function the header declares is defined in the stub (the stub
  includes the header, so the compiler already checks their signatures)
header_check.py HEADER --cyclops CYCLOPS_HEADER
  the copy declares Cyclops' functions with the same signatures, its
  constants have Cyclops' values, and its structs Cyclops' fields
"""

import re
import sys


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


def check_stub(header, stub):
  defined = set(re.findall(r"^[\w ]+?\*? ?(cyclops_\w+)\(", stub, re.M))
  return [
      f"{name} is declared but the stub does not define it"
      for name in sorted(functions(header).keys() - defined)
  ]


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
  header_path, mode, other_path = argv[1:]
  with open(header_path) as f:
    header = f.read()
  with open(other_path) as f:
    other = f.read()
  check = check_stub if mode == "--stub" else check_cyclops
  errors = check(header, other)
  if not functions(header):
    errors.append("no functions found in the header")
  for error in errors:
    print(error)
  return 1 if errors else 0


if __name__ == "__main__":
  sys.exit(main(sys.argv))
