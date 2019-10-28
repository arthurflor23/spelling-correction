"""
Provides options via the command line to perform project tasks.
* `--dataset`: dataset name (bea2019, bentham, conll13, conll14, google, iam, rimes, saintgall, washington)
* `--transform`: transform dataset to the corpus, sentences (train and test) files
* `--mode`: method to be used:

    `symspell`:
        * `--N`: max edit distance (2 by default)

    `luong`, `bahdanau`, `transformer`:
        * `--train`: train the model
        * `--test`: predict and evaluate sentences
        * `--epochs`: number of epochs
        * `--batch_size`: number of batches
"""

import os
import time
import argparse

from data import preproc as pp, evaluation
from data.generator import DataGenerator
from data.reader import Dataset

from tool.seq2seq import Seq2SeqAttention
from tool.symspell import Symspell
from tool.transformer import Transformer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--transform", action="store_true", default=False)
    parser.add_argument("--mode", type=str, default="bahdanau")
    parser.add_argument("--N", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    args = parser.parse_args()

    raw_path = os.path.join("..", "raw")
    data_path = os.path.join("..", "data")
    output_path = os.path.join("..", "output", args.dataset, args.mode)
    source = os.path.join(data_path, f"{args.dataset}.txt")

    max_text_length = 128
    charset_base = "".join([chr(i) for i in range(32, 127)])
    charset_special = "".join([chr(i) for i in range(192, 255)])

    if args.transform:
        names = next(os.walk(raw_path))[1]

        if args.dataset == "all":
            pass
        elif args.dataset == "htr":
            names = [x for x in names if x in ["bentham", "iam", "rimes", "saintgall", "washington"]]
        else:
            names = [args.dataset]

        dataset = Dataset(raw_path, names)
        dataset.read_lines(max_text_length)

        valid_noised = pp.add_noise(dataset.partitions["valid"], max_text_length)
        test_noised = pp.add_noise(dataset.partitions["test"], max_text_length)

        current_metric = evaluation.ocr_metrics(test_noised, dataset.partitions["test"])

        info = "\n".join([
            f"####",
            f"#### {args.dataset} partitions (number of sentences)",
            f"#### Total:      {dataset.total}",
            f"####\n",
            f"#### Train:      {dataset.total_train}",
            f"#### Validation: {dataset.total_valid}",
            f"#### Test:       {dataset.total_test}\n",
            f"#### Current Error Rate:",
            f"#### Test CER: {current_metric[0]:.8f}",
            f"#### Test WER: {current_metric[1]:.8f}\n"
        ])

        print(info, f"\n{args.dataset} transformed dataset is saving...")
        os.makedirs(data_path, exist_ok=True)

        with open(source, "w") as f:
            f.write(f"{info}\n\n")

            for item in dataset.partitions["train"]:
                f.write(f"TR_L {item}\n")

            for item, noise in zip(dataset.partitions["valid"], valid_noised):
                f.write(f"VA_L {item}\nVA_P {noise}\n")

            for item, noise in zip(dataset.partitions["test"], test_noised):
                f.write(f"TE_L {item}\nTE_P {noise}\n")

    else:
        os.makedirs(output_path, exist_ok=True)

        dtgen = DataGenerator(source=source,
                              batch_size=args.batch_size,
                              charset=(charset_base + charset_special),
                              max_text_length=max_text_length)

        if args.mode == "symspell":
            sspell = Symspell(output_path, args.N)
            train_corpus = sspell.load(corpus=dtgen.dataset["train"]["gt"])

            with open(os.path.join(output_path, "train.txt"), "w") as lg:
                print(train_corpus)
                lg.write(train_corpus)

            start_time = time.time()
            predicts = sspell.autocorrect(batch=dtgen.dataset["test"]["dt"])
            total_time = time.time() - start_time

            old_metric = evaluation.ocr_metrics(dtgen.dataset["test"]["dt"], dtgen.dataset["test"]["gt"])
            new_metric = evaluation.ocr_metrics(predicts, dtgen.dataset["test"]["gt"])

            pred_corpus, eval_corpus = evaluation.report(dtgen, predicts, [old_metric, new_metric], total_time,
                                                         plus=f"Max Edit distance:\t{args.N}\n")

            with open(os.path.join(output_path, "predict.txt"), "w") as lg:
                lg.write("\n".join(pred_corpus))
                print("\n".join(pred_corpus[:30]))

            with open(os.path.join(output_path, "evaluate.txt"), "w") as lg:
                lg.write(eval_corpus)
                print(eval_corpus)

        else:
            if args.mode == "transformer":
                dtgen.one_hot_process = False
                model = Transformer(num_layers=2, units=2048, d_model=256, num_heads=2, dropout=0.1, tokenizer=dtgen.tokenizer)
            else:
                model = Seq2SeqAttention(arch=args.mode, units=512, dropout=0.1, tokenizer=dtgen.tokenizer)

            # set parameter `learning_rate` to customize or get default value of the model
            model.compile()

            checkpoint = "checkpoint_weights.hdf5"
            model.load_checkpoint(target=os.path.join(output_path, checkpoint))

            if args.train:
                model.summary(output_path, "summary.txt")
                callbacks = model.get_callbacks(logdir=output_path, hdf5=checkpoint, verbose=1)

                start_time = time.time()
                h = model.fit_generator(generator=dtgen.next_train_batch(),
                                        epochs=args.epochs,
                                        steps_per_epoch=dtgen.train_steps,
                                        validation_data=dtgen.next_valid_batch(),
                                        validation_steps=dtgen.valid_steps,
                                        callbacks=callbacks,
                                        shuffle=True,
                                        verbose=1)
                total_time = time.time() - start_time

                loss = h.history['loss']
                accuracy = h.history['accuracy']

                val_loss = h.history['val_loss']
                val_accuracy = h.history['val_accuracy']

                time_epoch = (total_time / len(accuracy))
                total_item = (dtgen.total_train + dtgen.total_valid)
                best_epoch_index = val_accuracy.index(max(val_accuracy))

                train_corpus = "\n".join([
                    f"Total train sentences:      {dtgen.total_train}",
                    f"Total validation sentences: {dtgen.total_valid}",
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
                    lg.write(train_corpus)
                    print(train_corpus)

            elif args.test:
                start_time = time.time()
                predicts = model.predict_generator(generator=dtgen.next_test_batch(),
                                                   steps=dtgen.test_steps,
                                                   use_multiprocessing=True,
                                                   verbose=1)
                total_time = time.time() - start_time

                old_metric = evaluation.ocr_metrics(dtgen.dataset["test"]["dt"], dtgen.dataset["test"]["gt"])
                new_metric = evaluation.ocr_metrics(predicts, dtgen.dataset["test"]["gt"])

                pred_corpus, eval_corpus = evaluation.report(dtgen, predicts, [old_metric, new_metric], total_time)

                with open(os.path.join(output_path, "predict.txt"), "w") as lg:
                    lg.write("\n".join(pred_corpus))
                    print("\n".join(pred_corpus[:30]))

                with open(os.path.join(output_path, "evaluate.txt"), "w") as lg:
                    lg.write(eval_corpus)
                    print(eval_corpus)
