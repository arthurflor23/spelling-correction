<img src="https://github.com/arthurflor23/text-correction/blob/master/doc/image/header.png?raw=true">

A text corrector system implemented using the Statistical Language Model ([SymSpell](https://github.com/mammothb/symspellpy)) and Neural Network ([Seq2Seq](https://towardsdatascience.com/seq2seq-model-in-tensorflow-ec0c557e560f) and [Transformer](https://www.tensorflow.org/tutorials/text/transformer)) (using TensorFlow 2.0). This project supports several text datasets and uses a noise function to create more data training (unlike Grammatical Error Correction (GEC) methodology). Don't worry, this is an automatic process in `transform` step and generator class.

**Notes**:
1. All **references** are commented in the code.
2. Check out the presentation in the **doc** folder.
3. For more information and demo run step by step (neural network approach), check out the **[tutorial](https://github.com/arthurflor23/text-correction/blob/master/src/tutorial.ipynb)** on Google Colab/Drive.

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
* `--mode`: method to be used:

  `symspell`:

    * `--N`: max edit distance (2 by default)

  `luong`, `bahdanau`, `transformer`:

    * `--train`: train the model

    * `--test`: predict and evaluate sentences

    * `--epochs`: number of epochs

    * `--batch_size`: number of batches

## Tutorial (Google Colab/Drive)

A Jupyter Notebook is available to demo run (neural network approach), check out the **[tutorial](https://github.com/arthurflor23/text-correction/blob/master/src/tutorial.ipynb)** on Google Colab/Drive.