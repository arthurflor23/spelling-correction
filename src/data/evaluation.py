"""
Tool to metrics calculation through data and label (string and string).
 * Calculation from Optical Character Recognition (OCR) metrics with editdistance.
"""

import editdistance


def ocr_metrics(predict, ground_truth):
    """Calculate Character Error Rate (CER), Word Error Rate (WER) and Sequence Error Rate (SER)"""

    cer, wer, ser = [], [], []

    for (pd, gt) in zip(predict, ground_truth):
        pd_cer, gt_cer = list(pd.lower()), list(gt.lower())
        dist = editdistance.eval(pd_cer, gt_cer)
        cer.append(dist / (max(len(pd_cer), len(gt_cer))))

        pd_wer, gt_wer = pd.lower().split(), gt.lower().split()
        dist = editdistance.eval(pd_wer, gt_wer)
        wer.append(dist / (max(len(pd_wer), len(gt_wer))))

        pd_ser, gt_ser = [pd], [gt]
        dist = editdistance.eval(pd_ser, gt_ser)
        ser.append(dist / (max(len(pd_ser), len(gt_ser))))

    cer_f = sum(cer) / len(cer)
    wer_f = sum(wer) / len(wer)
    ser_f = sum(ser) / len(ser)

    return (cer_f, wer_f, ser_f)


def report(dtgen, new_dt, metrics, total_time, plus=""):
    """Calculate and organize metrics and predicts informations"""

    e_corpus = "\n".join([
        f"Total test sentences: {dtgen.size['test']}",
        f"{plus}",
        f"Total time:           {total_time:.8f} sec",
        f"Time per item:        {(total_time / dtgen.size['test']):.8f} sec\n",
        f"Metrics (before):",
        f"Character Error Rate: {metrics[0][0]:.8f}",
        f"Word Error Rate:      {metrics[0][1]:.8f}",
        f"Sequence Error Rate:  {metrics[0][2]:.8f}\n",
        f"Metrics (after):",
        f"Character Error Rate: {metrics[1][0]:.8f}",
        f"Word Error Rate:      {metrics[1][1]:.8f}",
        f"Sequence Error Rate:  {metrics[1][2]:.8f}"
    ])

    p_corpus = []
    for i in range(dtgen.size['test']):
        p_corpus.append(f"GT {dtgen.dataset['test']['gt'][i]}")
        p_corpus.append(f"DT {dtgen.dataset['test']['dt'][i]}")
        p_corpus.append(f"PD {new_dt[i]}\n")

    return (p_corpus, e_corpus)
