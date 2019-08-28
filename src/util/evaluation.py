"""
Tool to metrics calculation through label and predict (string | string).
 * Calculation from Optical Character Recognition (OCR) metrics with editdistance.
"""

import editdistance


def ocr_metrics(predict, label):
    """Calculate Character Error Rate (CER) and Word Error Rate (WER)"""

    cer, wer = [], []

    for (pred, lab) in zip(predict, label):
        pd, lb = list(pred), list(lab)
        dist = editdistance.eval(pd, lb)
        cer.append(dist / (max(len(pd), len(lb))))

        pd, lb = pred.split(), lab.split()
        dist = editdistance.eval(pd, lb)
        wer.append(dist / (max(len(pd), len(lb))))

    cer_f = sum(cer) / len(cer)
    wer_f = sum(wer) / len(wer)

    return (cer_f, wer_f)
