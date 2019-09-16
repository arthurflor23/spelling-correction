"""
Provides options via the command line to perform project tasks.
* `--dataset`: dataset name (bea2019, bentham, conll13, conll14, google, iam, rimes, washington)
* `--transform`: transform dataset to the corpus, sentences (train and test) files
* `--mode`: method to be used (symspell or network)
    `symspell`:
        * `--N`: 2 by default
    `network`:
        * `--method`: if network mode: type of model (seq2seq, gcnn, gcnn_ctc)
        * `--train`: if network mode: train the model (neural network)
        * `--test`: if network mode: evaluate and predict sentences
        * `--epochs`: if network mode: number of epochs
        * `--batch_size`: if network mode: number of batches
"""

import os
import time
import string
import argparse
import importlib

from data import preproc as pp, evaluation
from network import callbacks
from statistical import symspell
from data.generator import DataGenerator
from contextlib import redirect_stdout


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--transform", action="store_true", default=False)
    parser.add_argument("--mode", type=str, default="symspell")
    parser.add_argument("--N", type=int, default=2)
    parser.add_argument("--method", type=str, default="seq2seq")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    args = parser.parse_args()

    raw_path = os.path.join("..", "raw")
    data_path = os.path.join("..", "data", args.dataset)
    output_path = os.path.join("..", "output", args.dataset, args.mode)
    m2_src = os.path.join(data_path, f"{args.dataset}.m2")

    max_text_length = 128
    charset_base = string.printable[:95]
    charset_special = "ÀÁÂÃÅÇÈÉÊËÍÎÏÒÓÔÖÚÜàáâãäåçèéêëìíîïñòóôõöùúûüý"

    if args.transform:
        dataset_list = next(os.walk(raw_path))[1] if args.dataset == "all" else [args.dataset]
        train, valid, test = [], [], []

        for dataset in dataset_list:
            source_path = os.path.join(raw_path, dataset)
            assert os.path.exists(source_path)

            try:
                mod = importlib.import_module(f"transform.{dataset}")
                print(f"The {dataset} dataset will be transformed...")

                tfm = mod.Transform(source=source_path,
                                    charset=(charset_base + charset_special),
                                    max_text_length=max_text_length)
                tfm.build()
                train += tfm.partitions["train"]
                valid += tfm.partitions["valid"]
                test += tfm.partitions["test"]

            except Exception:
                print(f"The {dataset} dataset not found...")
                pass

        train = pp.shuffle(train)
        valid = pp.shuffle(valid)
        test = pp.shuffle(test)

        info = "\n".join([
            f"{args.dataset} partitions (number of sentences)",
            f"Train:        {len(train)}",
            f"Validation:   {len(valid)}",
            f"Test:         {len(test)}\n"
        ])

        print(f"\n{info}")
        print(f"{args.dataset} transformed dataset is saving...")
        os.makedirs(data_path, exist_ok=True)

        with open(m2_src, "w") as f:
            f.write("TR_L " + "\nTR_L ".join(train) + "\n")

            for item in valid:
                f.write(f"VA_L {item}\n")
                f.write(f"VA_P {pp.add_noise([item], max_text_length)[0]}\n")

            for item in test:
                f.write(f"TE_L {item}\n")
                f.write(f"TE_P {pp.add_noise([item], max_text_length)[0]}\n")

        with open(os.path.join(data_path, "about.txt"), "w") as f:
            f.write(info)

        print(f"Transformation finished.")

    else:
        dtgen = DataGenerator(m2_src=m2_src,
                              batch_size=args.batch_size,
                              charset=(charset_base + charset_special),
                              max_text_length=max_text_length,
                              ctc=("ctc" in args.method))

        if args.mode == "symspell":
            os.makedirs(output_path, exist_ok=True)
            sspell = symspell.Symspell(output_path, args.N)
            train_corpus = sspell.load(corpus=dtgen.dataset["train"]["gt"])

            with open(os.path.join(output_path, "train.txt"), "w") as lg:
                print(train_corpus)
                lg.write(train_corpus)

            start_time = time.time()
            predicts = sspell.autocorrect(batch=dtgen.dataset["test"]["dt"])
            total_time = time.time() - start_time

            old_metric = evaluation.ocr_metrics(dtgen.dataset["test"]["dt"], dtgen.dataset["test"]["gt"])
            new_metric = evaluation.ocr_metrics(predicts, dtgen.dataset["test"]["gt"])

            eval_corpus, pred_corpus = evaluation.report(dtgen, predicts, [old_metric, new_metric], total_time,
                                                         plus=f"Max Edit distance:\t{args.N}\n")

            with open(os.path.join(output_path, "evaluate.txt"), "w") as lg:
                print(eval_corpus)
                lg.write(eval_corpus)

            with open(os.path.join(output_path, "predict.txt"), "w") as lg:
                lg.write(pred_corpus)

        elif args.mode == "network":
            output_path = output_path.replace(args.mode, args.method)
            os.makedirs(output_path, exist_ok=True)

            nn = importlib.import_module(f"network.{args.method}")

            checkpoint = "checkpoint_weights.hdf5"
            cbs = callbacks.setup(logdir=output_path, hdf5_target=checkpoint)

            model = nn.generate_model(max_text_length=dtgen.max_text_length,
                                      charset_length=len(dtgen.charset),
                                      checkpoint=os.path.join(output_path, checkpoint))
            if args.train:
                with open(os.path.join(output_path, "summary.txt"), "w") as f:
                    model.summary()
                    with redirect_stdout(f):
                        model.summary()

                start_time = time.time()
                h = model.fit_generator(generator=dtgen.next_train_batch(),
                                        epochs=args.epochs,
                                        steps_per_epoch=dtgen.train_steps,
                                        validation_data=dtgen.next_valid_batch(),
                                        validation_steps=dtgen.valid_steps,
                                        callbacks=cbs,
                                        shuffle=True,
                                        verbose=1)
                total_time = time.time() - start_time

                loss = h.history['loss']
                val_loss = h.history['val_loss']

                min_val_loss = min(val_loss)
                min_val_loss_i = val_loss.index(min_val_loss)

                train_corpus = "\n".join([
                    f"Total train sentences:      {dtgen.total_train}",
                    f"Total validation sentences: {dtgen.total_valid}",
                    f"Batch:                      {dtgen.batch_size}\n",
                    f"Total time:                 {total_time:.4f} sec",
                    f"Average time per epoch:     {(total_time / len(loss)):.4f} sec\n",
                    f"Total epochs:               {len(loss)}",
                    f"Best epoch                  {min_val_loss_i + 1}\n",
                    f"Training loss:              {loss[min_val_loss_i]:.4f}",
                    f"Validation loss:            {min_val_loss:.4f}"
                ])

                with open(os.path.join(output_path, "train.txt"), "w") as lg:
                    print(train_corpus)
                    lg.write(train_corpus)

            elif args.test:
                start_time = time.time()
                predicts = model.predict_generator(generator=dtgen.next_test_batch(),
                                                   steps=dtgen.test_steps,
                                                   use_multiprocessing=True,
                                                   verbose=1)
                total_time = time.time() - start_time

                if not dtgen.ctc:
                    predicts = [pp.decode_onehot(x, dtgen.charset) for x in predicts]

                old_metric = evaluation.ocr_metrics(dtgen.dataset["test"]["dt"], dtgen.dataset["test"]["gt"])
                new_metric = evaluation.ocr_metrics(predicts, dtgen.dataset["test"]["gt"])

                eval_corpus, pred_corpus = evaluation.report(dtgen, predicts, [old_metric, new_metric], total_time)

                with open(os.path.join(output_path, "evaluate.txt"), "w") as lg:
                    print(eval_corpus)
                    lg.write(eval_corpus)

                with open(os.path.join(output_path, "predict.txt"), "w") as lg:
                    lg.write(pred_corpus)
