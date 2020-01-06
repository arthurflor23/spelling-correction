"""
Sequence to Sequence Network with Attention (seq2seq)

References:
    Ilya Sutskever and Oriol Vinyals and Quoc V. Le
    Sequence to Sequence Learning with Neural Networks, 2014
    arXiv, URL: https://arxiv.org/abs/1409.3215

    Wojciech Zaremba and Ilya Sutskever
    Learning to Execute, 2014
    arXiv, URL: http://arxiv.org/abs/1410.4615
"""

import os
import numpy as np
import tensorflow as tf

from contextlib import redirect_stdout
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Progbar, GeneratorEnqueuer

from tensorflow.keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Concatenate, Attention, AdditiveAttention, LayerNormalization
from tensorflow.keras.layers import Input, Bidirectional, GRU, TimeDistributed, Dense


class Seq2SeqAttention():
    """
    Seq2Seq with Attention implementation.
    This class implement Luong and Bahdanau architectures with layer normalization approach.
    """

    def __init__(self, tokenizer, mode, units, dropout=0.0):
        self.tokenizer = tokenizer
        self.mode = mode
        self.units = units
        self.dropout = dropout

        self.model = None
        self.encoder = None
        self.decoder = None

    def summary(self, output=None, target=None):
        """Show/Save model structure (summary)"""

        self.model.summary()

        if target is not None:
            os.makedirs(output, exist_ok=True)

            with open(os.path.join(output, target), "w") as f:
                with redirect_stdout(f):
                    self.model.summary()

    def load_checkpoint(self, target):
        """Restore model to construct the encoder and decoder"""

        if os.path.isfile(target):
            if self.model is None:
                self.compile()

            self.model.load_weights(target, by_name=True)
            self.encoder.load_weights(target, by_name=True)
            self.decoder.load_weights(target, by_name=True)

    def get_callbacks(self, logdir, checkpoint, monitor="val_loss", verbose=0):
        """Setup the list of callbacks for the model"""

        callbacks = [
            CSVLogger(
                filename=os.path.join(logdir, "epochs.log"),
                separator=";",
                append=True),
            TensorBoard(
                log_dir=logdir,
                histogram_freq=10,
                profile_batch=0,
                write_graph=True,
                write_images=False,
                update_freq="epoch"),
            ModelCheckpoint(
                filepath=checkpoint,
                monitor=monitor,
                save_best_only=True,
                save_weights_only=True,
                verbose=verbose),
            EarlyStopping(
                monitor=monitor,
                min_delta=1e-8,
                patience=20,
                restore_best_weights=True,
                verbose=verbose),
            ReduceLROnPlateau(
                monitor=monitor,
                min_delta=1e-8,
                factor=0.2,
                patience=15,
                verbose=verbose)
        ]

        return callbacks

    def compile(self, learning_rate=None):
        """
        Build models (train, encoder and decoder)

        Architecture based on Bahdanau and Transformer model approach.
            Reference:
                Dzmitry Bahdanau and Kyunghyun Cho and Yoshua Bengio
                Neural Machine Translation by Jointly Learning to Align and Translate, 2014
                arXiv, URL: https://arxiv.org/abs/1409.0473

        Architecture based on Luong and Transformer model approach.
            Reference:
                Minh-Thang Luong and Hieu Pham and Christopher D. Manning
                Effective Approaches to Attention-based Neural Machine Translation, 2015
                arXiv, URL: https://arxiv.org/abs/1508.04025

        More References:
            Thushan Ganegedara
            Attention in Deep Networks with Keras
            Medium: https://towardsdatascience.com/light-on-math-ml-attention-with-keras-dc8dbc1fad39
            Github: https://github.com/thushv89/attention_keras

            Trung Tran
            Neural Machine Translation With Attention Mechanism
            Machine Talk: https://machinetaltf.keras.backend.org/2019/03/29/neural-machine-translation-with-attention-mechanism/
            Github: https://github.com/ChunML/NLP/tree/master/machine_translation
        """

        # Encoder and Decoder Inputs
        encoder_inputs = Input(shape=(None, self.tokenizer.vocab_size), name="encoder_inputs")
        decoder_inputs = Input(shape=(None, self.tokenizer.vocab_size), name="decoder_inputs")

        # Encoder bgru
        encoder_bgru = Bidirectional(GRU(self.units, return_sequences=True, return_state=True,
                                         dropout=self.dropout), name="encoder_bgru")

        encoder_out, state_h, state_c = encoder_bgru(encoder_inputs)

        # Set up the decoder GRU, using `encoder_states` as initial state.
        decoder_gru = GRU(self.units * 2, return_sequences=True, return_state=True,
                          dropout=self.dropout, name="decoder_gru")

        decoder_out, _ = decoder_gru(decoder_inputs, initial_state=Concatenate(axis=-1)([state_h, state_c]))

        # Attention layer
        if self.mode == "bahdanau":
            attn_layer = AdditiveAttention(use_scale=False, name="attention_layer")
        else:
            attn_layer = Attention(use_scale=False, name="attention_layer")

        attn_out = attn_layer([decoder_out, encoder_out])

        # Normalization layer
        norm_layer = LayerNormalization(name="normalization")
        decoder_concat_input = norm_layer(Concatenate(axis=-1)([decoder_out, attn_out]))

        # Dense layer
        dense = Dense(self.tokenizer.vocab_size, activation="softmax", name="softmax_layer")
        dense_time_distributed = TimeDistributed(dense, name="time_distributed_layer")

        decoder_pred = dense_time_distributed(decoder_concat_input)

        """ Train model """
        if learning_rate is None:
            learning_rate = 0.001

        optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0, clipvalue=0.5, epsilon=1e-8)

        self.model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred, name="seq2seq")
        self.model.compile(optimizer=optimizer, loss=self.loss_func, metrics=['accuracy'])

        """ Inference model """

        """ Encoder (Inference) model """
        self.encoder = Model(inputs=encoder_inputs, outputs=[encoder_out, state_h, state_c])

        """ Decoder (Inference) model """
        # Decoder Inputs (states)
        encoder_inf_states = Input(shape=(self.tokenizer.maxlen, self.units * 2), name="encoder_inf_states")
        decoder_init_states = Input(shape=(self.units * 2), name="decoder_init")
        decoder_inf_inputs = Input(shape=(1, self.tokenizer.vocab_size), name="decoder_inf_inputs")

        # Decoder GRU
        decoder_inf_out, decoder_inf_states = decoder_gru(decoder_inf_inputs, initial_state=decoder_init_states)

        # Attention layer
        attn_inf_out = attn_layer([decoder_inf_out, encoder_inf_states])

        # Normalization layer
        decoder_inf_concat = norm_layer(Concatenate(axis=-1)([decoder_inf_out, attn_inf_out]))

        # Dense layer
        decoder_inf_pred = dense_time_distributed(decoder_inf_concat)

        # Decoder model
        self.decoder = Model(inputs=[encoder_inf_states, decoder_init_states, decoder_inf_inputs],
                             outputs=[decoder_inf_pred, decoder_inf_states])

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.0,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            **kwargs):
        """
        Model training on data yielded (fit function has support to generator).
        A fit() abstration function of TensorFlow 2 using the model_train.

        :param: See tensorflow.keras.Model.fit()
        :return: A history object
        """

        out = self.model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs, verbose=verbose,
                             callbacks=callbacks, validation_split=validation_split,
                             validation_data=validation_data, shuffle=shuffle,
                             class_weight=class_weight, sample_weight=sample_weight,
                             initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch,
                             validation_steps=validation_steps, validation_freq=validation_freq,
                             max_queue_size=max_queue_size, workers=workers,
                             use_multiprocessing=use_multiprocessing, **kwargs)
        return out

    def predict(self,
                x,
                batch_size=None,
                verbose=0,
                steps=1,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False):
        """
        Model predicting on data yielded (generator).
        A predict() abstration function of TensorFlow 2 using the encoder and decoder models

        :param: See tensorflow.keras.Model.predict()
        :return: A numpy array(s) of predictions.

        References:
            Tal Weiss
            Deep Spelling
            Medium: https://machinelearnings.co/deep-spelling-9ffef96a24f6
            Github: https://github.com/MajorTal/DeepSpell

            Vu Tran
            Sequence-to-Sequence Learning for Spelling Correction
            Github: https://github.com/vuptran/deep-spell-checkr
        """

        self.encoder._make_predict_function()
        self.decoder._make_predict_function()

        try:
            enqueuer = GeneratorEnqueuer(x, use_multiprocessing=use_multiprocessing)
            enqueuer.start(workers=workers, max_queue_size=max_queue_size)
            output_generator = enqueuer.get()

            steps_done = 0
            if verbose == 1:
                print("Model Predict")
                progbar = Progbar(target=steps)

            predicts = []

            while steps_done < steps:
                x = next(output_generator)[0]
                batch_size = len(x)

                # Encode the input as state vectors
                encoder_out, state_h, state_c = self.encoder.predict(x)
                dec_state = tf.concat([state_h, state_c], axis=-1)

                # Create batch of empty target sequences of length 1 character and populate
                # the first element of target sequence with the # start-of-sequence character
                target = np.zeros((batch_size, 1, self.tokenizer.vocab_size))
                target[:, 0, self.tokenizer.SOS] = 1.0

                # Sampling loop for a batch of sequences
                decoded_tokens = [''] * batch_size

                for _ in range(self.tokenizer.maxlen):
                    # `char_probs` has shape (batch_size, 1, nb_target_chars)
                    char_probs, dec_state = self.decoder.predict([encoder_out, dec_state, target])

                    # Reset the target sequences.
                    target = np.zeros((batch_size, 1, self.tokenizer.vocab_size))

                    # Sample next character using argmax or multinomial mode
                    sampled_chars = []

                    for i in range(batch_size):
                        next_index = char_probs[i].argmax(axis=-1)
                        next_char = self.tokenizer.decode([next_index])

                        decoded_tokens[i] += next_char
                        sampled_chars.append(next_char)

                        # Update target sequence with index of next character
                        target[i, 0, next_index] = 1.0

                    stop_char = set(sampled_chars)

                    if len(stop_char) == 1 and stop_char.pop() == self.tokenizer.EOS_TK:
                        break

                # Sampling finished
                predicts.extend([self.tokenizer.remove_tokens(x) for x in decoded_tokens])

                steps_done += 1
                if verbose == 1:
                    progbar.update(steps_done)

        finally:
            enqueuer.stop()

        return predicts

    @staticmethod
    def loss_func(y_true, y_pred):
        """Loss function with CategoryCrossentropy and label smoothing"""

        return tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1, reduction="none")(y_true, y_pred)
