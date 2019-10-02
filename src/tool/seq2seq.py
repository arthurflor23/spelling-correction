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
from tensorflow.keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Concatenate, Bidirectional, GRU
from tensorflow.keras.layers import TimeDistributed, Dense, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence, GeneratorEnqueuer, OrderedEnqueuer, Progbar


class Seq2SeqAttention():
    """
    Seq2Seq with Attention implementation.
    This class model has the Bahdanau approach arch with the Luong-style attention.

    References:
        Minh-Thang Luong and Hieu Pham and Christopher D. Manning
        Effective Approaches to Attention-based Neural Machine Translation, 2015
        arXiv, URL: https://arxiv.org/abs/1508.04025

        Dzmitry Bahdanau and Kyunghyun Cho and Yoshua Bengio
        Neural Machine Translation by Jointly Learning to Align and Translate, 2014
        arXiv, URL: https://arxiv.org/abs/1409.0473
    """

    def __init__(self, units, dropout, tokenizer):
        self.units = units
        self.dropout = dropout
        self.tokenizer = tokenizer

        self.model = None
        self.encoder = None
        self.decoder = None

    def summary(self, output=None, target=None):
        """Show/Save model structure (summary)"""

        if target is not None:
            os.makedirs(output, exist_ok=True)

            with open(os.path.join(output, target), "w") as f:
                with redirect_stdout(f):
                    self.model.summary()
        self.model.summary()

    def load_checkpoint(self, target):
        """Restore model to construct the encoder and decoder"""

        if os.path.isfile(target):
            if self.model is None:
                self.compile()

            self.model.load_weights(target, by_name=True)
            self.encoder.load_weights(target, by_name=True)
            self.decoder.load_weights(target, by_name=True)

    def get_callbacks(self, logdir, hdf5, monitor="val_accuracy", verbose=0):
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
                filepath=os.path.join(logdir, hdf5),
                monitor=monitor,
                save_best_only=True,
                save_weights_only=True,
                verbose=verbose),
            EarlyStopping(
                monitor=monitor,
                min_delta=0,
                patience=20,
                restore_best_weights=True,
                verbose=verbose),
            ReduceLROnPlateau(
                monitor=monitor,
                min_delta=0,
                factor=0.2,
                patience=10,
                verbose=verbose)
        ]

        return callbacks

    def compile(self, learning_rate=0.001):
        """
        Build models (train, encoder and decoder)

        References:
            Thushan Ganegedara
            Attention in Deep Networks with Keras
            Medium: https://towardsdatascience.com/light-on-math-ml-attention-with-keras-dc8dbc1fad39
            Github: https://github.com/thushv89/attention_keras

            Trung Tran
            Neural Machine Translation With Attention Mechanism
            Machine Talk: https://machinetalk.org/2019/03/29/neural-machine-translation-with-attention-mechanism/
            Github: https://github.com/ChunML/NLP/tree/master/machine_translation
        """

        # Encoder and Decoder Inputs
        encoder_inputs = Input(shape=(None, self.tokenizer.vocab_size), name="encoder_inputs")
        decoder_inputs = Input(shape=(None, self.tokenizer.vocab_size), name="decoder_inputs")

        # Encoder BiGRU
        encoder_bigru = Bidirectional(GRU(self.units, return_sequences=True, return_state=True,
                                          name="encoder_gru", dropout=self.dropout), name="encoder_bigru")
        encoder_out, encoder_fwd_state, encoder_back_state = encoder_bigru(encoder_inputs)
        encoder_states = Concatenate(axis=-1)([encoder_fwd_state, encoder_back_state])

        # Set up the decoder GRU, using `encoder_states` as initial state.
        decoder_gru = GRU(self.units * 2, return_sequences=True, return_state=True,
                          name="decoder_gru", dropout=self.dropout)
        decoder_out, decoder_state = decoder_gru(decoder_inputs, initial_state=encoder_states)

        # Attention layer
        attn_layer = Attention(name="attention_layer")
        attn_out = attn_layer([decoder_out, encoder_out])
        decoder_concat_input = Concatenate(axis=-1)([decoder_out, attn_out])

        # Dense layer
        dense = Dense(self.tokenizer.vocab_size, activation="softmax", name="softmax_layer")
        dense_time = TimeDistributed(dense, name="time_distributed_layer")
        decoder_pred = dense_time(decoder_concat_input)

        if learning_rate is None:
            learning_rate = CustomSchedule(self.tokenizer.vocab_size)

        optimizer = Adam(learning_rate=learning_rate)

        # Full model
        self.model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred, name="seq2seq")
        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

        """ Inference model """

        """ Encoder (Inference) model """
        # Encoder model
        self.encoder = Model(inputs=encoder_inputs, outputs=[encoder_out, encoder_fwd_state, encoder_back_state])

        """ Decoder (Inference) model """
        # Decoder Inputs (states)
        decoder_inf_inputs = Input(shape=(1, self.tokenizer.vocab_size), name="decoder_word_inputs")
        encoder_inf_states = Input(shape=(self.tokenizer.maxlen, self.units * 2), name="encoder_inf_states")
        decoder_init_state = Input(shape=(self.units * 2), name="decoder_init")

        # Decoder GRU
        decoder_inf_out, decoder_inf_state = decoder_gru(decoder_inf_inputs, initial_state=decoder_init_state)

        # Attention layer
        attn_inf_out = attn_layer([decoder_inf_out, encoder_inf_states])
        decoder_inf_concat = Concatenate(axis=-1)([decoder_inf_out, attn_inf_out])

        # Dense layer
        decoder_inf_pred = TimeDistributed(dense)(decoder_inf_concat)

        # Decoder model
        self.decoder = Model(inputs=[encoder_inf_states, decoder_init_state, decoder_inf_inputs],
                             outputs=[decoder_inf_pred, decoder_inf_state])

    def fit_generator(self,
                      generator,
                      steps_per_epoch,
                      epochs=1,
                      verbose=1,
                      callbacks=None,
                      validation_data=None,
                      validation_steps=None,
                      class_weight=None,
                      max_queue_size=10,
                      workers=1,
                      shuffle=True,
                      initial_epoch=0):
        """
        Model training on data yielded batch-by-batch by a Python generator.

        The generator is run in parallel to the model, for efficiency.
        For instance, this allows you to do real-time data augmentation on images on CPU in parallel to training your model on GPU.

        A major modification concerns the generator that must provide x data of the form:
          [input_sequences_encoder, input_sequences_decoder, label_sequences]

        :param: See tensorflow.keras.engine.Model.fit_generator()
        :return: A History object
        """
        out = self.model.fit_generator(generator, steps_per_epoch, epochs=epochs, verbose=verbose,
                                       callbacks=callbacks, validation_data=validation_data,
                                       validation_steps=validation_steps, class_weight=class_weight,
                                       max_queue_size=max_queue_size, workers=workers, shuffle=shuffle,
                                       initial_epoch=initial_epoch)
        return out

    def predict_generator(self,
                          generator,
                          steps,
                          max_queue_size=10,
                          workers=1,
                          use_multiprocessing=False,
                          verbose=0):
        """
        Generates predictions for the input samples from a data generator.
        The generator should return the same kind of data as accepted by `predict_on_batch`.

        :param: See tensorflow.keras.engine.Model.predict_generator()
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
        is_sequence = isinstance(generator, Sequence)

        steps_done = 0
        enqueuer = None

        try:
            if is_sequence:
                enqueuer = OrderedEnqueuer(generator, use_multiprocessing=use_multiprocessing)
            else:
                enqueuer = GeneratorEnqueuer(generator, use_multiprocessing=use_multiprocessing)

            enqueuer.start(workers=workers, max_queue_size=max_queue_size)
            output_generator = enqueuer.get()
            predicts = []

            if verbose == 1:
                progbar = Progbar(target=steps)

            while steps_done < steps:
                x = next(output_generator)
                batch_size = len(x[0])

                # Encode the input as state vectors
                encoder_out, state_h, state_c = self.encoder.predict(x)
                dec_state = np.concatenate([state_h, state_c], axis=-1)

                # Create batch of empty target sequences of length 1 character and populate
                # the first element of target sequence with the # start-of-sequence character
                target_sequences = np.zeros((batch_size, 1, self.tokenizer.vocab_size))
                target_sequences[:, 0, self.tokenizer.SOS] = 1.0

                # Sampling loop for a batch of sequences
                decoded_tokens = [""] * batch_size

                for _ in range(self.tokenizer.maxlen):
                    # `char_probs` has shape (batch_size, 1, nb_target_chars)
                    char_probs, dec_state = self.decoder.predict([encoder_out, dec_state, target_sequences])

                    # Reset the target sequences.
                    target_sequences = np.zeros((batch_size, 1, self.tokenizer.vocab_size))

                    # Sample next character using argmax or multinomial mode
                    sampled_chars = []

                    for i in range(batch_size):
                        next_index = char_probs[i].argmax(axis=-1)
                        next_char = self.tokenizer.decode([next_index])

                        decoded_tokens[i] += next_char
                        sampled_chars.append(next_char)

                        # Update target sequence with index of next character
                        target_sequences[i, 0, next_index] = 1.0

                    stop_char = set(sampled_chars)
                    if len(stop_char) == 1 and stop_char.pop() == self.tokenizer.EOS_TK:
                        break

                # Sampling finished
                predicts.extend([self.tokenizer.remove_tokens(x) for x in decoded_tokens])

                steps_done += 1
                if verbose == 1:
                    progbar.update(steps_done)

        finally:
            if enqueuer is not None:
                enqueuer.stop()

        return predicts


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Custom schedule of the learning rate with warmup_steps.
    From original paper "Attention is all you need".
    """

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
