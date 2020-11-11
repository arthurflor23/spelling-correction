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

## Samples

Samples using the output of an HTR system (corresponds to 'Baseline')

### Bentham

<img src="https://github.com/arthurflor23/spelling-correction/blob/master/doc/samples/bentham.png?raw=true">

### IAM

<img src="https://github.com/arthurflor23/spelling-correction/blob/master/doc/samples/iam.png?raw=true">

### RIMES

<img src="https://github.com/arthurflor23/spelling-correction/blob/master/doc/samples/rimes.png?raw=true">

### Saint Gall

<img src="https://github.com/arthurflor23/spelling-correction/blob/master/doc/samples/saintgall.png?raw=true">

### Washington

<img src="https://github.com/arthurflor23/spelling-correction/blob/master/doc/samples/washington.png?raw=true">

## Citation

If this project helped in any way in your research work, feel free to cite the following papers.

### HTR-Flor++: A Handwritten Text Recognition System Based on a Pipeline of Optical and Language Models ([here](https://doi.org/10.1145/3395027.3419603))

This work aimed to propose a different pipeline for Handwritten Text Recognition (HTR) systems in post-processing, using two steps to correct the output text. The first step aimed to correct the text at the character level (using N-gram model). The second step had the objective of correcting the text at the word level (using a word frequency dictionary). The experiment was validated in the IAM dataset and compared to the best works proposed within this data scenario.

```
@inproceedings{10.1145/3395027.3419603,
    author      = {Neto, Arthur F. S. and Bezerra, Byron L. D. and Toselli, Alejandro H. and Lima, Estanislau B.},
    title       = {{HTR-Flor++:} A Handwritten Text Recognition System Based on a Pipeline of Optical and Language Models},
    booktitle   = {Proceedings of the ACM Symposium on Document Engineering 2020},
    year        = {2020},
    publisher   = {Association for Computing Machinery},
    address     = {New York, NY, USA},
    location    = {Virtual Event, CA, USA},
    series      = {DocEng '20},
    isbn        = {9781450380003},
    url         = {https://doi.org/10.1145/3395027.3419603},
    doi         = {10.1145/3395027.3419603},
}
```

### Towards the Natural Language Processing as Spelling Correction for Offline Handwritten Text Recognition Systems ([here](https://doi.org/10.3390/app10217711))

This work aimed a deep study within the research field of Natural Language Processing (NLP), and to bring its approaches to the research field of Handwritten Text Recognition (HTR). Thus, for the experiment and validation, we used 5 datasets (Bentham, IAM, RIMES, Saint Gall and Washington), 3 optical models (Bluche, Puigcerver, Flor), and 8 techniques for text correction in post-processing, including approaches statistics and neural networks, such as encoder-decoder models (seq2seq and Transformers).

```
@article{10.3390/app10217711,
    author  = {Neto, Arthur F. S. and Bezerra, Byron L. D. and Toselli, Alejandro H.},
    title   = {Towards the Natural Language Processing as Spelling Correction for Offline Handwritten Text Recognition Systems},
    journal = {Applied Sciences},
    pages   = {1-29},
    month   = {10},
    year    = {2020},
    volume  = {10},
    number  = {21},
    url     = {https://doi.org/10.3390/app10217711},
    doi     = {10.3390/app10217711},
}
```
