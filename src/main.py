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
import string
import argparse
import importlib

from data import preproc
from util import evaluation
from data.generator import DataGenerator
from symspellpy.symspellpy import SymSpell


def main(args):
    raw_dir = os.path.join("..", "raw")
    data_dir = os.path.join("..", "data", args.dataset)
    output_dir = os.path.join("..", "output", args.dataset, args.mode)

    m2_src = os.path.join(data_dir, f"{args.dataset}.m2")
    max_text_length = 128
    charset_base = "".join([chr(i) for i in range(32, 127)])
    charset_special = "".join([chr(i) for i in range(192, 256)])

    if args.transform:
        dataset_list = next(os.walk(raw_dir))[1] if args.dataset == "all" else [args.dataset]
        train, valid, test = [], [], []

        for dataset in dataset_list:
            source_dir = os.path.join(raw_dir, dataset)
            assert os.path.exists(source_dir)

            try:
                mod = importlib.import_module(f"transform.{dataset}")
                print(f"The {dataset} dataset will be transformed...")

                tfm = mod.Transform(source=source_dir,
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
            f"{args.dataset} partition (sentences)",
            f"Train:        {len(train)}",
            f"Validation:   {len(valid)}",
            f"Test:         {len(test)}\n"
        ])

        print(f"\n{info}")
        print(f"{dataset} transformed dataset is saving...")
        os.makedirs(data_dir, exist_ok=True)

        with open(m2_src, "w") as f:
            f.write("TR_L " + "\nTR_L ".join(train) + "\n")

            for item in valid:
                f.write(f"VA_L {item}\n")
                f.write(f"VA_P {preproc.add_noise(item)[0]}\n")

            for item in test:
                f.write(f"TE_L {item}\n")
                f.write(f"TE_P {preproc.add_noise(item)[0]}\n")

        with open(os.path.join(data_dir, "about.txt"), "w") as f:
            f.write(info)

        print(f"Transformation finished.")

    elif args.mode == "symspell":
        os.makedirs(output_dir, exist_ok=True)
        corpus = os.path.join(output_dir, "corpus.data")
        dictionary = os.path.join(output_dir, "dictionary.data")

        sym_spell = SymSpell(max_dictionary_edit_distance=args.max_distance, prefix_length=7)
        dtgen = DataGenerator(m2_src=m2_src, batch_size=args.batch_size, max_text_length=max_text_length)

        if not os.path.isfile(dictionary):
            start_time = time.time()

            with open(corpus, "w") as f:
                f.write(preproc.parse_sentence(" ".join(dtgen.dataset["train"]["gt"])))
            sym_spell.create_dictionary(corpus)

            with open(dictionary, "w") as f:
                for key, count in sym_spell.words.items():
                    f.write(f"{key} {count}\n")

            total_time = time.time() - start_time

            train_corpus = "\n".join([
                f"Total train images:   {dtgen.total_train}",
                f"Total time:           {total_time:.4f} sec",
                f"Time per item:        {(total_time / dtgen.total_train):.4f} sec\n",
            ])

            with open(os.path.join(output_dir, "train.txt"), "w") as lg:
                print(train_corpus)
                lg.write(train_corpus)

        sym_spell.load_dictionary(dictionary, term_index=0, count_index=1)
        start_time = time.time()
        new_dt = []

        for i in range(dtgen.total_test):
            corrected = []

            for item in dtgen.dataset["test"]["dt"][i].split():
                temp = item
                if item not in string.punctuation:
                    sugg = sym_spell.lookup(item, "TOP", args.max_edit_distance, transfer_casing=True)
                    temp = sugg[0].term if sugg else item

                corrected.append(preproc.organize_space(temp))
            new_dt.append(" ".join(corrected))

        total_time = time.time() - start_time
        eval_corpus, pred_corpus = evaluation.report(dtgen, new_dt, total_time)

        with open(os.path.join(output_dir, "evaluate.txt"), "w") as lg:
            print(eval_corpus)
            lg.write(eval_corpus)

        with open(os.path.join(output_dir, "predict.txt"), "w") as lg:
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
