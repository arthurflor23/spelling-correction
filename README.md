# Text Correction

A Text corrector system implemented using [SymSpell](https://github.com/mammothb/symspellpy) (N-gram) and [TensorFlow 2.0](https://www.tensorflow.org/) (Neural Network). This project supports several text datasets, however it has a different methodology for training and validating all data, when compared to the Grammar Error Correction (GEC) methodology. Don't worry, this all is an automatic process in `transform` step.

This project has a statistical language model with an N-gram approach and a Neural Network approach, both with the purpose of text correction (simple approach to the spell checker).

**Notes**:
1. All **references** are commented in the code.
2. This project presents its own partitioning and training methodology.
3. Check out the presentation in the **doc** folder.
4. For more information and demo run step by step, check out the **[tutorial](https://github.com/arthurflor23/text-correction/blob/master/src/tutorial.ipynb)** on Google Colab/Drive.

## Datasets supported

a. [CoNLL13](https://www.comp.nus.edu.sg/~nlp/conll13st.html)

b. [CoNLL14](https://www.comp.nus.edu.sg/~nlp/conll14st.html)

c. [BEA2019](https://www.cl.cam.ac.uk/research/nl/bea2019st/)

d. [Bentham](http://transcriptorium.eu/datasets/bentham-collection/)

e. [Google](https://ai.google/research/pubs/pub41880)

f. [IAM](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)

g. [Rimes](http://www.a2ialab.com/doku.php?id=rimes_database:start)

h. [Washington](http://www.fki.inf.unibe.ch/databases/iam-historical-document-database/washington-database)

## Requirements

* Python 3.x
* Symspellpy
* editdistance
* TensorFlow 2.0

## Command line arguments

* `--dataset`: dataset name (conll13, conll14, bea2019, bentham, google, iam, rimes, washington)
* `--transform`: transform dataset to the m2 file
* `--train`: train model using the dataset argument
* `--test`: evaluate and predict model using the dataset argument
* `--mode`: execute on `ngram` or `neuralnetwork` mode

## Tutorial (Google Colab/Drive)

A Jupyter Notebook is available to demo run, check out the **[tutorial](https://github.com/arthurflor23/text-correction/blob/master/src/tutorial.ipynb)** on Google Colab/Drive.