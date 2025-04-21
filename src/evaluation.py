import json
import argparse
from radgraph import F1RadGraph
from f1chexbert import F1CheXbert
from rouge import Rouge
import evaluate
from bert_score import BERTScorer

def main(args):
    with open(args.ref_path, "r") as f:
        ref_dict = json.load(f)
        
    with open(args.pred_path, "r") as f:
        pred_dict = json.load(f)
    print(f"References size: {len(ref_dict)}, Preditions size: {len(pred_dict)}")
    hyps = []
    refs = []
    for ref, pred in zip(ref_dict, pred_dict):
        hyp = pred['retrieved_finding'][0]
        check = [" ".join(_.split()) for _ in hyp.split(".") if len(_) > 0]
        if len(check) > 0:
            hyps.append(pred['retrieved_finding'][0])
            refs.append(ref['finding'])

    assert len(hyps) == len(refs)
    print("The number of testing dataset:", len(hyps))

    # F1RadGraph Evaluation
    f1radgraph = F1RadGraph(reward_level=args.radgraph_level, model_type="radgraph")
    score, _, _, _ = f1radgraph(hyps=hyps, refs=refs)
    print("F1RadGraph:", score)

    # F1CheXbert Evaluation
    f1chexbert = F1CheXbert(device=args.device)
    _, _, _, class_report_5 = f1chexbert(hyps=hyps, refs=refs)
    print("F1CheXpert:", class_report_5["micro avg"]["f1-score"])

    # ROUGE Evaluation
    rouge = Rouge()
    scores = rouge.get_scores(hyps, refs, avg=True)
    print("Rouge-L:", scores['rouge-l']['f'])

    # BLEU Evaluation
    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=hyps, references=[[ref] for ref in refs])
    print("BLEU-4 score:", results['precisions'][3])

    # BERTScore Evaluation
    bert_scorer = BERTScorer(model_type=args.bert_model,
                             num_layers=5,
                             batch_size=64,
                             nthreads=4,
                             all_layers=False,
                             idf=False,
                             device=args.device,
                             lang='en',
                             rescale_with_baseline=True,
                             baseline_path=None)
    _, _, f = bert_scorer.score(cands=hyps, refs=refs)
    print("BERTScore:", f.mean().item())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_path", type=str, required=True, help="Path to reference JSON file")
    parser.add_argument("--pred_path", type=str, required=True, help="Path to predicted JSON file")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device (cpu or cuda)")
    parser.add_argument("--radgraph_level", type=str, default="partial", help="RadGraph reward level")
    parser.add_argument("--bert_model", type=str, default="distilbert-base-uncased", help="BERTScore model type")
    
    args = parser.parse_args()
    main(args)
