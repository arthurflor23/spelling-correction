"""
Tool to metrics calculation through data and label (string | string).
 * Calculation from Optical Character Recognition (OCR) metrics with editdistance.
"""

import editdistance


def ocr_metrics(predict, ground_truth):
    """Calculate Character Error Rate (CER) and Word Error Rate (WER)"""

    cer, wer = [], []

    for (pd, gt) in zip(predict, ground_truth):
        pd_cer, gt_cer = list(pd.lower()), list(gt.lower())
        dist = editdistance.eval(pd_cer, gt_cer)
        cer.append(dist / (max(len(pd_cer), len(gt_cer))))

        pd_wer, gt_wer = pd.lower().split(), gt.lower().split()
        dist = editdistance.eval(pd_wer, gt_wer)
        wer.append(dist / (max(len(pd_wer), len(gt_wer))))

    cer_f = sum(cer) / len(cer)
    wer_f = sum(wer) / len(wer)

    return (cer_f, wer_f)


def report(dtgen, new_dt, metrics, total_time, plus=""):
    """Calculate and organize metrics and predicts informations"""

    eval_corpus = "\n".join([
        f"Total test sentences: {dtgen.total_test}",
        f"{plus}",
        f"Total time:           {total_time:.4f} sec",
        f"Time per item:        {(total_time / dtgen.total_test):.4f} sec\n",
        f"Metrics (before):",
        f"Character Error Rate: {metrics[0][0]:.4f}",
        f"Word Error Rate:      {metrics[0][1]:.4f}\n",
        f"Metrics (after):",
        f"Character Error Rate: {metrics[1][0]:.4f}",
        f"Word Error Rate:      {metrics[1][1]:.4f}"
    ])

    pred_corpus = ""
    for i in range(dtgen.total_test):
        pred_corpus += f"GT {dtgen.dataset['test']['gt'][i]}\n"
        pred_corpus += f"DT {dtgen.dataset['test']['dt'][i]}\n"
        pred_corpus += f"PD {new_dt[i]}\n\n"

    return (eval_corpus, pred_corpus)
