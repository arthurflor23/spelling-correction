"""
Provides options via the command line to perform project tasks.
* `--source`: dataset/model name (bea2019, bentham, conll13, conll14, google, iam, rimes, saintgall, washington)
* `--transform`: transform dataset to the standard project file
* `--mode`: method to be used:

    `similarity`, `norvig`, `symspell`:
        * `--N`: N gram or max edit distance (2 by default)
        * `--train`: create corpus files
        * `--test`: predict and evaluate sentences

    `luong`, `bahdanau`, `transformer`:
        * `--train`: train the model
        * `--test`: predict and evaluate sentences
        * `--epochs`: number of epochs
        * `--batch_size`: number of batches
"""

import argparse
import os
import string
import time

from data import preproc as pp, evaluation as ev
from data.generator import DataGenerator
from data.reader import Dataset

from tool.seq2seq import Seq2SeqAttention
from tool.statistical import LanguageModel
from tool.transformer import Transformer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="all")
    parser.add_argument("--transform", action="store_true", default=False)
    parser.add_argument("--mode", type=str, default="bahdanau")

    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
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
        names = next(os.walk(raw_path))[1]

        if args.source == "all":
            pass
        elif args.source == "htr":
            names = [x for x in names if x in ['bentham', 'iam', 'rimes', 'saintgall', 'washington']]
        else:
            names = [args.source]

        dataset = Dataset(source=raw_path, names=names)
        dataset.read_lines(maxlen=max_text_length)

        valid_noised = pp.add_noise(dataset.partitions['valid'], max_text_length)
        test_noised = pp.add_noise(dataset.partitions['test'], max_text_length)

        current_metric = ev.ocr_metrics(test_noised, dataset.partitions['test'])

        info = "\n".join([
            f"####",
            f"#### {args.source} partitions (number of sentences)",
            f"#### Total:      {dataset.size['total']}",
            f"####\n",
            f"#### Train:      {dataset.size['train']}",
            f"#### Validation: {dataset.size['valid']}",
            f"#### Test:       {dataset.size['test']}\n",
            f"#### Current Error Rate:",
            f"#### Test CER: {current_metric[0]:.8f}",
            f"#### Test WER: {current_metric[1]:.8f}\n"
        ])

        print(info, f"\n{args.source} transformed dataset is saving...")
        os.makedirs(data_path, exist_ok=True)

        with open(source_path, "w") as f:
            f.write(f"{info}\n\n")

            for item in dataset.partitions['train']:
                f.write(f"TR_L {item}\n")

            for item, noise in zip(dataset.partitions['valid'], valid_noised):
                f.write(f"VA_L {item}\nVA_P {noise}\n")

            for item, noise in zip(dataset.partitions['test'], test_noised):
                f.write(f"TE_L {item}\nTE_P {noise}\n")

    else:
        assert os.path.isfile(source_path) or os.path.isfile(target_path)
        os.makedirs(output_path, exist_ok=True)

        dtgen = DataGenerator(source=source_path,
                              batch_size=args.batch_size,
                              charset=(charset_base + charset_special),
                              max_text_length=max_text_length,
                              predict=args.test)

        if args.mode in ['kaldi', 'similarity', 'norvig', 'symspell']:
            lm = LanguageModel(mode=args.mode, source=source_path, N=args.N)

            if args.train:
                if args.mode == "kaldi":
                    print("\n##########################################\n")
                    print("You'll have to work hard for this option.\n")
                    print("See some instructions in the ``src/tool/statistical.py`` file (kaldi function section)")
                    print("and also in the ``src/tool/lib/kaldi-srilm-script.sh`` file. \n☘️ ☘️ ☘️")
                    print("\n##########################################\n")
                else:
                    corpus = lm.create_corpus(sentences=dtgen.dataset['train']['gt'])

                    with open(os.path.join(output_path, "corpus.txt"), "w") as lg:
                        lg.write(corpus)

            elif args.test:
                if args.mode != "kaldi":
                    lm.read_corpus(corpus_path=os.path.join(output_path, "corpus.txt"))

                start_time = time.time()
                predicts = lm.autocorrect(sentences=dtgen.dataset['test']['dt'])
                total_time = time.time() - start_time

                old_metric = ev.ocr_metrics(dtgen.dataset['test']['dt'], dtgen.dataset['test']['gt'])
                new_metric = ev.ocr_metrics(predicts, dtgen.dataset['test']['gt'])

                p_corpus, e_corpus = ev.report(dtgen, predicts, [old_metric, new_metric], total_time, plus=f"N: {args.N}\n")

                with open(os.path.join(output_path, "predict.txt"), "w") as lg:
                    lg.write("\n".join(p_corpus))
                    print("\n".join(p_corpus[:30]))

                with open(os.path.join(output_path, "evaluate.txt"), "w") as lg:
                    lg.write(e_corpus)
                    print(e_corpus)

        else:
            if args.mode == "transformer":
                dtgen.one_hot_process = False
                model = Transformer(dtgen.tokenizer, num_layers=4, units=1024, d_model=128, num_heads=8, dropout=0.1)
            else:
                model = Seq2SeqAttention(dtgen.tokenizer, args.mode, units=1024, dropout=0.1)

            # set parameter `learning_rate` to customize or get default value
            model.compile(learning_rate=0.001)
            model.load_checkpoint(target=target_path)

            if args.train:
                model.summary(output_path, "summary.txt")
                callbacks = model.get_callbacks(logdir=output_path, checkpoint=target_path, verbose=1)

                start_time = time.time()
                h = model.fit(x=dtgen.next_train_batch(),
                              epochs=args.epochs,
                              steps_per_epoch=dtgen.steps['train'],
                              validation_data=dtgen.next_valid_batch(),
                              validation_steps=dtgen.steps['valid'],
                              callbacks=callbacks,
                              shuffle=True,
                              verbose=1)
                total_time = time.time() - start_time

                loss = h.history['loss']
                accuracy = h.history['accuracy']

                val_loss = h.history['val_loss']
                val_accuracy = h.history['val_accuracy']

                time_epoch = (total_time / len(accuracy))
                total_item = (dtgen.size['train'] + dtgen.size['valid'])
                best_epoch_index = val_accuracy.index(max(val_accuracy))

                t_corpus = "\n".join([
                    f"Total train sentences:      {dtgen.size['train']}",
                    f"Total validation sentences: {dtgen.size['valid']}",
                    f"Batch:                      {dtgen.batch_size}\n",
                    f"Total epochs:               {len(accuracy)}",
                    f"Total time:                 {(total_time / 60):.2f} min",
                    f"Time per epoch:             {(time_epoch / 60):.2f} min",
                    f"Time per item:              {(time_epoch / total_item):.8f} sec\n",
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
                start_time = time.time()
                predicts = model.predict(x=dtgen.next_test_batch(),
                                         steps=dtgen.steps['test'],
                                         verbose=1)
                total_time = time.time() - start_time

                old_metric = ev.ocr_metrics(dtgen.dataset['test']['dt'], dtgen.dataset['test']['gt'])
                new_metric = ev.ocr_metrics(predicts, dtgen.dataset['test']['gt'])

                p_corpus, e_corpus = ev.report(dtgen, predicts, [old_metric, new_metric], total_time)

                with open(os.path.join(output_path, "predict.txt"), "w") as lg:
                    lg.write("\n".join(p_corpus))
                    print("\n".join(p_corpus[:30]))

                with open(os.path.join(output_path, "evaluate.txt"), "w") as lg:
                    lg.write(e_corpus)
                    print(e_corpus)
