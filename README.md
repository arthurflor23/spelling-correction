# Text Correction

A Text corrector system implemented using [SymSpell](https://github.com/mammothb/symspellpy) (Edit Distance) and [TensorFlow 2.0](https://www.tensorflow.org/) (Neural Network). This project supports several text datasets, however it has a different methodology for training and validating all data, when compared to the Grammar Error Correction (GEC) methodology. Don't worry, this all is an automatic process in `transform` step.

This project has a Statistical Language Model approach and a Neural Network approach, both with the purpose of text correction (simple approach to the spell checker).

**Notes**:
1. All **references** are commented in the code.
2. This project presents its own partitioning and training methodology.
3. Check out the presentation in the **doc** folder.
4. For more information and demo run step by step (neural network), check out the **[tutorial](https://github.com/arthurflor23/text-correction/blob/master/src/tutorial.ipynb)** on Google Colab/Drive.

## Datasets supported

a. [BEA2019](https://www.cl.cam.ac.uk/research/nl/bea2019st/)

b. [Bentham](http://transcriptorium.eu/datasets/bentham-collection/)

c. [CoNLL13](https://www.comp.nus.edu.sg/~nlp/conll13st.html)

d. [CoNLL14](https://www.comp.nus.edu.sg/~nlp/conll14st.html)

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

## Tutorial (Google Colab/Drive)

A Jupyter Notebook is available to demo run (neural network), check out the **[tutorial](https://github.com/arthurflor23/text-correction/blob/master/src/tutorial.ipynb)** on Google Colab/Drive.