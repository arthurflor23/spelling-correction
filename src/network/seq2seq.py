"""
Sequence to Sequence Network (seq2seq)

References:
    Ilya Sutskever and Oriol Vinyals and Quoc V. Le
    Sequence to Sequence Learning with Neural Networks, 2014
    arXiv
    URL: https://arxiv.org/abs/1409.3215

    Wojciech Zaremba and Ilya Sutskever
    Learning to Execute, 2014
    arXiv
    URL: http://arxiv.org/abs/1410.4615
"""

import os
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector
from tensorflow.keras.layers import TimeDistributed, Dense, Activation


def generate_model(max_text_length, charset_length, checkpoint=None):
    """Generate seq2seq model"""

    input_data = Input(name="input", shape=(max_text_length, charset_length))

    encoder = LSTM(units=512, return_sequences=True, dropout=0.5)(input_data)
    encoder = LSTM(units=512, return_sequences=False, dropout=0.5)(encoder)

    decoder = RepeatVector(max_text_length)(encoder)

    decoder = LSTM(units=512, return_sequences=True, dropout=0.5)(decoder)
    decoder = LSTM(units=512, return_sequences=True, dropout=0.5)(decoder)

    output_data = TimeDistributed(Dense(units=charset_length))(decoder)
    output_data = Activation(activation="softmax")(output_data)

    model = Model(inputs=input_data, outputs=output_data)
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

    if checkpoint and os.path.isfile(checkpoint):
        model.load_weights(checkpoint)

    return model
