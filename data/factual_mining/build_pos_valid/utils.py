
def exact_entity_token_if_rel_exists_reward(
        hypothesis_annotation_list, reference_annotation_list
):
  candidates = []
  for annotation_list in [hypothesis_annotation_list, reference_annotation_list]:
    candidate = []
    for entity in annotation_list.values():
      if not entity["relations"]:
        candidate.append((entity["tokens"], entity["label"]))
      if entity["relations"]:
        candidate.append((entity["tokens"], entity["label"], True))
    candidate = set(candidate)
    candidates.append(candidate)
  hypothesis_relation_token_list, reference_relation_token_list = candidates
  precision = (
      sum(
          [
              1
              for x in hypothesis_relation_token_list
              if (x in reference_relation_token_list)
          ]
      )
      / len(hypothesis_relation_token_list)
      if len(hypothesis_relation_token_list) > 0
      else 0.0
  )
  recall = (
      sum(
          [
              1
              for x in reference_relation_token_list
              if (x in hypothesis_relation_token_list)
          ]
      )
      / len(reference_relation_token_list)
      if len(reference_relation_token_list) > 0
      else 0.0
  )
  f1_score = (
      (2 * precision * recall / (precision + recall))
      if (precision + recall) > 0
      else 0.0
  )
  return f1_score


def chexbert_similarity(report, ret_report):
  report_label = report["label"]
  ret_report_label = ret_report["label"]
  # distance = manhattan_distance(report_label, ret_report_label)
  # # Calculate the similarity as the inverse of the distance plus one
  # # to prevent division by zero when the distance is zero.
  # return int(report_label == ret_report_label)
  return sum(1 for true, pred in zip(report_label, ret_report_label) if true == pred) / len(report_label)


def radgraph_similarity(report, ret_report):
  report_entities = report["entities"]
  ret_report_entities = ret_report["entities"]
  partial_reward = exact_entity_token_if_rel_exists_reward(
    ret_report_entities, report_entities)
  return partial_reward