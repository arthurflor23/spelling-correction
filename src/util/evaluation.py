"""
Tool to metrics calculation through data and label (string | string).
 * Calculation from Optical Character Recognition (OCR) metrics with editdistance.
"""

import editdistance


def ocr_metrics(data, label):
    """Calculate Character Error Rate (CER) and Word Error Rate (WER)"""

    cer, wer = [], []

    for (pred, lab) in zip(data, label):
        pd, lb = list(pred.lower()), list(lab.lower())
        dist = editdistance.eval(pd, lb)
        cer.append(dist / (max(len(pd), len(lb))))

        pd, lb = pred.lower().split(), lab.lower().split()
        dist = editdistance.eval(pd, lb)
        wer.append(dist / (max(len(pd), len(lb))))

    cer_f = sum(cer) / len(cer)
    wer_f = sum(wer) / len(wer)

    return (cer_f, wer_f)


def report(dtgen, new_dt, total_time):
    """
    Calculate and organize metrics and predicts informations
    """

    old_metric = ocr_metrics(dtgen.dataset["test"]["dt"], dtgen.dataset["test"]["gt"])
    new_metric = ocr_metrics(new_dt, dtgen.dataset["test"]["gt"])
    pred_corpus = ""

    eval_corpus = "\n".join([
        f"Total test images:    {dtgen.total_test}",
        f"Total time:           {total_time:.4f} sec",
        f"Time per item:        {(total_time / dtgen.total_test):.4f} sec\n",
        f"Metrics (before):",
        f"Character Error Rate: {old_metric[0]:.4f}",
        f"Word Error Rate:      {old_metric[1]:.4f}\n",
        f"Metrics (after):",
        f"Character Error Rate: {new_metric[0]:.4f}",
        f"Word Error Rate:      {new_metric[1]:.4f}"
    ])

    for i in range(dtgen.total_test):
        pred_corpus += f"GT {dtgen.dataset['test']['gt'][i]}\n"
        pred_corpus += f"DT {dtgen.dataset['test']['dt'][i]}\n"
        pred_corpus += f"PD {new_dt[i]}\n\n"

    return (eval_corpus, pred_corpus)
