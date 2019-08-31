"""
Provides options via the command line to perform project tasks.
* `--dataset`: dataset name (bea2019, bentham, conll13, conll14, google, iam, rimes, washington)
* `--transform`: transform dataset to the corpus, sentences (train and test) files
* `--train`: create dictionary (n-gram) / train the model (neural network)
* `--test`: evaluate and predict sentences with the dataset and type argument
* `--type`: method to be used (n-gram or neural network)
"""

import os
import argparse
import importlib
from data import preproc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=False, default="all")
    parser.add_argument("--transform", action="store_true", default=False)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--type", type=str, required=False, default="ngram")
    args = parser.parse_args()

    raw_dir = os.path.join("..", "raw")
    data_dir = os.path.join("..", "data", args.dataset)

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
        print(f"The {dataset} transformed is saving...")
        os.makedirs(data_dir, exist_ok=True)

        with open(os.path.join(data_dir, f"{args.dataset}.m2"), "w") as f:
            f.write("TR_L " + "\nTR_L ".join(train) + "\n")

            for item in valid:
                f.write(f"VA_L {item}\n")
                f.write(f"VA_S {preproc.add_noise(item)[0]}\n")

            for item in test:
                f.write(f"TE_L {item}\n")
                f.write(f"TE_S {preproc.add_noise(item)[0]}\n")

        with open(os.path.join(data_dir, "corpus.txt"), "w") as f:
            f.write(" ".join(train))

        with open(os.path.join(data_dir, "about.txt"), "w") as f:
            f.write(info)

        print(f"Transformation finished.")

    elif args.train:

        # make a n-gram option
        # corpus.txt to create dictionary and save

        # make a generator option to train neural network
        import util.m2 as m2
        dataset = m2.read_dataset(os.path.join(data_dir, f"{args.dataset}.m2"))

        print(dataset["train"]["lab"][0])
        print(dataset["train"]["sen"][0])
