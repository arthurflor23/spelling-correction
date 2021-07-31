"""
Provides options via the command line to perform project tasks.
* `--source`: dataset/model name (bea2019, bentham, conll13, conll14, google, iam, rimes, saintgall, washington)
* `--transform`: transform dataset to the standard project file
* `--mode`: method to be used:

    `similarity`, `norvig`, `symspell`:
        * `--N`: N gram or max edit distance
        * `--train`: create corpus files
        * `--test`: predict and evaluate sentences

    `luong`, `bahdanau`, `transformer`:
        * `--train`: train the model
        * `--test`: predict and evaluate sentences
        * `--epochs`: number of epochs
        * `--batch_size`: number of batches

* `--norm_accentuation`: discard accentuation marks in the evaluation
* `--norm_punctuation`: discard punctuation marks in the evaluation
"""

import argparse
import os
import string
import datetime

from data import preproc as pp, evaluation as ev
from data.generator import DataGenerator
from data.reader import Dataset

from tool.seq2seq import Seq2SeqAttention
from tool.statistical import LanguageModel
from tool.transformer import Transformer


def report(dtgen, predicts, metrics, total_time, plus=""):
    """Calculate and organize metrics and predicts informations"""

    e_corpus = "\n".join([
        f"Total test sentences: {dtgen.size['test']}",
        f"{plus}",
        f"Total time:           {total_time}",
        f"Time per item:        {total_time / dtgen.size['test']}\n",
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
        p_corpus.append(f"PD {predicts[i]}\n")

    return (p_corpus, e_corpus)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="bea2019")
    parser.add_argument("--transform", action="store_true", default=False)
    parser.add_argument("--mode", type=str, default="luong")

    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)

    parser.add_argument("--norm_accentuation", action="store_true", default=False)
    parser.add_argument("--norm_punctuation", action="store_true", default=False)

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--N", type=int, default=2)
    args = parser.parse_args()

    raw_path = os.path.join("..", "raw")
    data_path = os.path.join("..", "data")
    source_path = os.path.join(data_path, f"{args.source}.txt")
    output_path = os.path.join("..", "output", args.source, args.mode)
    target_path = os.path.join(output_path, "checkpoint_weights.hdf5")

    max_text_length = 128
    charset_base = string.printable[:95]
    charset_special = """ÀÁÂÃÄÅÇÈÉÊËÌÍÎÏÑÒÓÔÕÖÙÚÛÜÝàáâãäåçèéêëìíîïñòóôõöùúûüý"""

    if args.transform:
        data = Dataset(source=os.path.join(raw_path, args.source))
        data.read_lines(maxlen=max_text_length)

        valid_noised = pp.add_noise(data.dataset['valid'], max_text_length)
        test_noised = pp.add_noise(data.dataset['test'], max_text_length)

        valid_metrics = ev.ocr_metrics(ground_truth=data.dataset['valid'], data=valid_noised)

        info = "\n".join([
            f"####",
            f"#### {args.source} partitions (number of sentences)",
            f"####",
            f"#### Total:      {data.size['total']}",
            f"####",
            f"#### Train:      {data.size['train']}",
            f"#### Validation: {data.size['valid']}",
            f"####\n",
            f"#### Validation Error Rate:",
            f"#### CER: {valid_metrics[0]:.8f}",
            f"#### WER: {valid_metrics[1]:.8f}",
            f"#### SER: {valid_metrics[2]:.8f}",
        ])

        print(info, f"\n{args.source} transformed dataset is saving...")
        os.makedirs(data_path, exist_ok=True)

        with open(source_path, "w") as f:
            f.write(f"{info}\n\n")

            for item in data.dataset['train']:
                f.write(f"TR_L {item}\n")

            for item, noise in zip(data.dataset['valid'], valid_noised):
                f.write(f"VA_L {item}\nVA_P {noise}\n")

            for item, noise in zip(data.dataset['test'], test_noised):
                f.write(f"TE_L {item}\nTE_P {noise}\n")

    else:
        assert os.path.isfile(source_path) or os.path.isfile(target_path)
        os.makedirs(output_path, exist_ok=True)

        dtgen = DataGenerator(source=source_path,
                              batch_size=args.batch_size,
                              charset=(charset_base + charset_special),
                              max_text_length=max_text_length,
                              predict=args.test)

        if args.mode in ['similarity', 'norvig', 'symspell', 'kaldi']:
            lm = LanguageModel(mode=args.mode, output=output_path, N=args.N)

            if args.train:
                if args.mode == "kaldi":
                    lm.autocorrect(sentences=None, predict=args.test)
                else:
                    corpus = lm.create_corpus(dtgen.dataset['train']['gt'] +
                                              dtgen.dataset['valid']['gt'] +
                                              dtgen.dataset['test']['gt'])

                    with open(os.path.join(output_path, "corpus.txt"), "w") as lg:
                        lg.write(corpus)

            elif args.test:
                if args.mode != "kaldi":
                    lm.read_corpus(corpus_path=os.path.join(output_path, "corpus.txt"))

                start_time = datetime.datetime.now()

                predicts = lm.autocorrect(sentences=dtgen.dataset['test']['dt'])
                predicts = [pp.text_standardize(x) for x in predicts]

                total_time = datetime.datetime.now() - start_time

                old_metric, new_metric = ev.ocr_metrics(ground_truth=dtgen.dataset['test']['gt'],
                                                        data=dtgen.dataset['test']['dt'],
                                                        predict=predicts,
                                                        norm_accentuation=args.norm_accentuation,
                                                        norm_punctuation=args.norm_punctuation)

                p_corpus, e_corpus = report(dtgen=dtgen,
                                            predicts=predicts,
                                            metrics=[old_metric, new_metric],
                                            total_time=total_time,
                                            plus=f"N: {args.N}\n")

                sufix = ("_norm" if args.norm_accentuation or args.norm_punctuation else "") + \
                        ("_accentuation" if args.norm_accentuation else "") + \
                        ("_punctuation" if args.norm_punctuation else "")

                with open(os.path.join(output_path, "predict.txt"), "w") as lg:
                    lg.write("\n".join(p_corpus))
                    print("\n".join(p_corpus[:30]))

                with open(os.path.join(output_path, f"evaluate{sufix}.txt"), "w") as lg:
                    lg.write(e_corpus)
                    print(e_corpus)

        else:
            if args.mode == "transformer":
                dtgen.one_hot_process = False
                model = Transformer(dtgen.tokenizer,
                                    num_layers=6,
                                    units=128,
                                    d_model=64,
                                    num_heads=8,
                                    dropout=0.1,
                                    stop_tolerance=20,
                                    reduce_tolerance=15)
            else:
                model = Seq2SeqAttention(dtgen.tokenizer,
                                         args.mode,
                                         units=128,
                                         dropout=0.2,
                                         stop_tolerance=20,
                                         reduce_tolerance=15)

            # set `learning_rate` parameter or None for custom schedule learning
            model.compile(learning_rate=0.001)
            model.load_checkpoint(target=target_path)

            if args.train:
                model.summary(output_path, "summary.txt")
                callbacks = model.get_callbacks(logdir=output_path, checkpoint=target_path, verbose=1)

                start_time = datetime.datetime.now()

                h = model.fit(x=dtgen.next_train_batch(),
                              epochs=args.epochs,
                              steps_per_epoch=dtgen.steps['train'],
                              validation_data=dtgen.next_valid_batch(),
                              validation_steps=dtgen.steps['valid'],
                              callbacks=callbacks,
                              shuffle=True,
                              verbose=1)

                total_time = datetime.datetime.now() - start_time

                loss = h.history['loss']
                accuracy = h.history['accuracy']

                val_loss = h.history['val_loss']
                val_accuracy = h.history['val_accuracy']

                time_epoch = (total_time / len(loss))
                total_item = (dtgen.size['train'] + dtgen.size['valid'])
                best_epoch_index = val_loss.index(min(val_loss))

                t_corpus = "\n".join([
                    f"Total train sentences:      {dtgen.size['train']}",
                    f"Total validation sentences: {dtgen.size['valid']}",
                    f"Batch:                      {dtgen.batch_size}\n",
                    f"Total epochs:               {len(accuracy)}",
                    f"Total time:                 {total_time}",
                    f"Time per epoch:             {time_epoch}",
                    f"Time per item:              {time_epoch / total_item}\n",
                    f"Best epoch                  {best_epoch_index + 1}",
                    f"Training loss:              {loss[best_epoch_index]:.8f}",
                    f"Training accuracy:          {accuracy[best_epoch_index]:.8f}\n",
                    f"Validation loss:            {val_loss[best_epoch_index]:.8f}",
                    f"Validation accuracy:        {val_accuracy[best_epoch_index]:.8f}"
                ])

                with open(os.path.join(output_path, "train.txt"), "w") as lg:
                    lg.write(t_corpus)
                    print(t_corpus)

            elif args.test:
                start_time = datetime.datetime.now()

                predicts = model.predict(x=dtgen.next_test_batch(), steps=dtgen.steps['test'], verbose=1)
                predicts = [pp.text_standardize(x) for x in predicts]

                total_time = datetime.datetime.now() - start_time

                old_metric, new_metric = ev.ocr_metrics(ground_truth=dtgen.dataset['test']['gt'],
                                                        data=dtgen.dataset['test']['dt'],
                                                        predict=predicts,
                                                        norm_accentuation=args.norm_accentuation,
                                                        norm_punctuation=args.norm_punctuation)

                p_corpus, e_corpus = report(dtgen=dtgen,
                                            predicts=predicts,
                                            metrics=[old_metric, new_metric],
                                            total_time=total_time)

                sufix = ("_norm" if args.norm_accentuation or args.norm_punctuation else "") + \
                        ("_accentuation" if args.norm_accentuation else "") + \
                        ("_punctuation" if args.norm_punctuation else "")

                with open(os.path.join(output_path, "predict.txt"), "w") as lg:
                    lg.write("\n".join(p_corpus))
                    print("\n".join(p_corpus[:30]))

                with open(os.path.join(output_path, f"evaluate{sufix}.txt"), "w") as lg:
                    lg.write(e_corpus)
                    print(e_corpus)
