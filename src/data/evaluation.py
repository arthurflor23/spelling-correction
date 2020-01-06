"""
Tool to metrics calculation through data and label (string and string).
 * Calculation from Optical Character Recognition (OCR) metrics with editdistance.
"""

import editdistance
import numpy as np

def ocr_metrics(predicts, ground_truth, outliers=True):
    """Calculate Character Error Rate (CER), Word Error Rate (WER) and Sequence Error Rate (SER)"""

    if len(predicts) == 0 or len(ground_truth) == 0:
        return (1, 1, 1)

    cer, wer, ser = [], [], []

    for (pd, gt) in zip(predicts, ground_truth):
        pd, gt = pd.lower(), gt.lower()

        pd_cer, gt_cer = list(pd), list(gt)
        dist = editdistance.eval(pd_cer, gt_cer)
        cer.append(dist / (max(len(pd_cer), len(gt_cer))))

        pd_wer, gt_wer = pd.split(), gt.split()
        dist = editdistance.eval(pd_wer, gt_wer)
        wer.append(dist / (max(len(pd_wer), len(gt_wer))))

        pd_ser, gt_ser = [pd], [gt]
        dist = editdistance.eval(pd_ser, gt_ser)
        ser.append(dist / (max(len(pd_ser), len(gt_ser))))

    if not outliers:
        mean, std = np.mean(cer), np.std(cer)
        no_outliers = [i for i in range(len(cer)) if (cer[i] > mean - 2 * std)]
        no_outliers = [i for i in no_outliers if (cer[i] < mean + 2 * std)]

        cer = [cer[i] for i in no_outliers]
        wer = [wer[i] for i in no_outliers]
        ser = [ser[i] for i in no_outliers]

    return (np.mean(cer), np.mean(wer), np.mean(ser))
