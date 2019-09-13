"""
Provides options via the command line to perform project tasks.
* `--dataset`: dataset name (bea2019, bentham, conll13, conll14, google, iam, rimes, washington)
* `--transform`: transform dataset to the corpus, sentences (train and test) files
* `--mode`: method to be used (symspell or neuralnetwork)
    `symspell`:
        * `--max_edit_distance`: 2 by default
    `neuralnetwork`:
        * `--train`: if neuralnetwork mode: train the model (neural network)
        * `--test`: if neuralnetwork mode: evaluate and predict sentences
        * `--epochs`: if neuralnetwork mode: number of epochs
        * `--batch_size`: if neuralnetwork mode: number of batches
"""

import os
import time
import argparse
import importlib

from data import preproc, evaluation
from tool.symspell import Symspell
from data.generator import DataGenerator


def main(args):
    raw_path = os.path.join("..", "raw")
    data_path = os.path.join("..", "data", args.dataset)
    output_path = os.path.join("..", "output", args.dataset, args.mode)

    m2_src = os.path.join(data_path, f"{args.dataset}.m2")
    max_text_length = 128
    charset_base = "".join([chr(i) for i in range(32, 127)])
    charset_special = "".join([chr(i) for i in range(192, 256)])

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
                del tfm

            except Exception:
                print(f"The {dataset} dataset not found...")
                pass

        info = "\n".join([
            f"{args.dataset} partition (number of sentences)",
            f"Train:        {len(train)}",
            f"Validation:   {len(valid)}",
            f"Test:         {len(test)}\n"
        ])

        print(f"\n{info}")
        print(f"{dataset} transformed dataset is saving...")
        os.makedirs(data_path, exist_ok=True)

        with open(m2_src, "w") as f:
            f.write("TR_L " + "\nTR_L ".join(train) + "\n")

            for item in valid:
                f.write(f"VA_L {item}\n")
                f.write(f"VA_P {preproc.add_noise(item, max_text_length)[0]}\n")

            for item in test:
                f.write(f"TE_L {item}\n")
                f.write(f"TE_P {preproc.add_noise(item, max_text_length)[0]}\n")

        with open(os.path.join(data_path, "about.txt"), "w") as f:
            f.write(info)

        print(f"Transformation finished.")

    else:
        os.makedirs(output_path, exist_ok=True)
        dtgen = DataGenerator(m2_src=m2_src,
                              batch_size=args.batch_size,
                              max_text_length=max_text_length,
                              charset=(charset_base + charset_special))

        if args.mode == "symspell":
            symspell = Symspell(output_path, args.max_edit_distance)
            train_corpus = symspell.load(corpus=dtgen.dataset["train"]["gt"])

            with open(os.path.join(output_path, "train.txt"), "w") as lg:
                print(train_corpus)
                lg.write(train_corpus)

            start_time = time.time()
            predict = symspell.autocorrect(batch=dtgen.dataset["test"]["dt"])
            total_time = time.time() - start_time

            old_metric = evaluation.ocr_metrics(dtgen.dataset["test"]["dt"], dtgen.dataset["test"]["gt"])
            new_metric = evaluation.ocr_metrics(predict, dtgen.dataset["test"]["gt"])

            eval_corpus, pred_corpus = evaluation.report(dtgen, predict, [old_metric, new_metric], total_time,
                                                         plus=f"Max Edit distance:\t{args.max_edit_distance}\n")

            with open(os.path.join(output_path, "evaluate.txt"), "w") as lg:
                print(eval_corpus)
                lg.write(eval_corpus)

            with open(os.path.join(output_path, "predict.txt"), "w") as lg:
                lg.write(pred_corpus)

        elif args.mode == "neuralnetwork":

            if args.train:
                print("train neural_network")

            elif args.test:
                print("test neural_network")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=False, default="all")
    parser.add_argument("--transform", action="store_true", default=False)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--mode", type=str, required=False, default="symspell")
    parser.add_argument("--max_edit_distance", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    main(args)
