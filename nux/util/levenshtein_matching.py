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

    matching[closest_target] = i
    targets.remove(closest_target)

  return matching
