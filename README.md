<img src="https://github.com/arthurflor23/spelling-correction/blob/master/doc/image/header.png?raw=true">

A spell corrector system implemented using the Statistical Language Model ([Ngram](https://github.com/gpoulter/python-ngram), [Pyspellchecker](https://github.com/barrust/pyspellchecker) and [SymSpell](https://github.com/mammothb/symspellpy)) and Neural Network ([Seq2Seq](https://towardsdatascience.com/seq2seq-model-in-tensorflow-ec0c557e560f) and [Transformer](https://www.tensorflow.org/tutorials/text/transformer)) with TensorFlow 2.x. This project supports several text datasets and uses a noise random function to create data training (unlike Grammatical Error Correction (GEC) methodology). Don't worry, this is an automatic process in `transform` step and generator class.

**Notes**:

1. All **references** are commented in the code.
2. Check out the presentation in the **doc** folder.
3. For more information and demo run step by step (neural network approach), check out the **[tutorial](https://github.com/arthurflor23/spelling-correction/blob/master/src/tutorial.ipynb)** on Google Colab/Drive.

## Datasets supported

a. [BEA2019](https://www.cl.cam.ac.uk/research/nl/bea2019st/)

b. [Bentham](http://transcriptorium.eu/datasets/bentham-collection/)

c. [CoNLL13](https://www.comp.nus.edu.sg/~nlp/conll13st.html)

d. [CoNLL14](https://www.comp.nus.edu.sg/~nlp/conll14st.html)

e. [Google](https://ai.google/research/pubs/pub41880)

f. [IAM](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)

g. [Rimes](http://www.a2ialab.com/doku.php?id=rimes_database:start)

h. [Saint Gall](http://www.fki.inf.unibe.ch/databases/iam-historical-document-database/saint-gall-database)

i. [Washington](http://www.fki.inf.unibe.ch/databases/iam-historical-document-database/washington-database)

## Requirements

- Python 3.x
- Editdistance
- Ngram
- Pyspellchecker
- Symspellpy
- TensorFlow 2.x

## Command line arguments

- `--source`: dataset/model name (bea2019, bentham, conll13, conll14, google, iam, rimes, saintgall, washington)
- `--transform`: transform dataset to the standard project file
- `--mode`: method to be used:

  `similarity`, `norvig`, `symspell`:

  - `--N`: N gram or max edit distance

  - `--train`: create corpus files

  - `--test`: predict and evaluate sentences

  `luong`, `bahdanau`, `transformer`:

  - `--train`: train the model

  - `--test`: predict and evaluate sentences

  - `--epochs`: number of epochs

  - `--batch_size`: number of batches

- `--norm_accentuation`: discard accentuation marks in the evaluation
- `--norm_punctuation`: discard punctuation marks in the evaluation

## Tutorial (Google Colab/Drive)

A Jupyter Notebook is available to demo run (neural network approach), check out the **[tutorial](https://github.com/arthurflor23/spelling-correction/blob/master/src/tutorial.ipynb)** on Google Colab/Drive.
