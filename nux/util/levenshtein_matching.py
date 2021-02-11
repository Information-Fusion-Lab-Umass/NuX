from fuzzywuzzy import fuzz

def match_strings(targets, inputs):
  # Using the Levenshtein distance
  assert len(targets) == len(inputs)
  targets = set(targets)

  matching = {}

  for i in inputs:
    closest_target, best_ratio = None, 0.0
    for t in targets:
      ratio = fuzz.ratio(t, i)
      if ratio > best_ratio:
        closest_target = t
        best_ratio = ratio
      elif ratio == best_ratio:
        import pdb; pdb.set_trace()

    matching[closest_target] = i
    targets.remove(closest_target)

  return matching

def len_common_prefix(x, y):
  prefix = "".join([a if a == b else "#" for a, b in zip(x, y)]).split("#")[0]
  return len(prefix)

def match_strings_using_shapes(targets, inputs, shape_map):
  # Using the Levenshtein distance
  assert len(targets) == len(inputs)
  targets = set(targets)

  matching = {}

  for i in inputs:
    closest_target, best_ratio = None, 0.0
    for t in targets:
      if shape_map[i] != shape_map[t]:
        continue
      ratio = fuzz.ratio(t, i)
      if ratio > best_ratio:
        closest_target = t
        best_ratio = ratio
      elif ratio == best_ratio:
        # Take the one with the shortest common prefix
        prefix1 = len_common_prefix(i, t)
        prefix2 = len_common_prefix(i, closest_target)
        if prefix1 < prefix2:
          closest_target = t
          best_ratio = ratio

    matching[closest_target] = i
    targets.remove(closest_target)

  return matching
