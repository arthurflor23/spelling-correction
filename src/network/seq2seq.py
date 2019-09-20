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
import numpy as np

from tensorflow.keras.utils import Sequence, GeneratorEnqueuer, OrderedEnqueuer, Progbar
from tensorflow.keras.layers import TimeDistributed, Dense, AdditiveAttention
from tensorflow.keras.layers import Input, Concatenate, Bidirectional, GRU, LSTM
from contextlib import redirect_stdout
from tensorflow.keras import Model


class Seq2SeqModel():

    def __init__(self, units, charset):
        self.units = units
        self.charset = charset
        self.SOS = charset[0]
        self.EOS = charset[-1]
        self.charset_size = len(charset)

        self.model_train = None
        self.model_encoder = None
        self.model_decoder = None

    def compile(self, optimizer="adam"):
        """Build and compile models (train, encoder and decoder)"""
        # # Define the main model consisting of encoder and decoder.
        # encoder_inputs = Input(shape=(None, self.charset_size), name='encoder_data')
        # encoder_lstm = LSTM(self.units, dropout=0.2, return_sequences=True, return_state=False, name='encoder_lstm_1')
        # encoder_outputs = encoder_lstm(encoder_inputs)

        # encoder_lstm = LSTM(self.units, dropout=0.2, return_sequences=False, return_state=True, name='encoder_lstm_2')
        # _, state_h, state_c = encoder_lstm(encoder_outputs)

        # # Set up the decoder, using `encoder_states` as initial state.
        # decoder_inputs = Input(shape=(None, self.charset_size), name='decoder_data')
        # # We set up our decoder to return full output sequences,
        # # and to return internal states as well. We don't use the return
        # # states in the training model, but we will use them in inference.
        # decoder_lstm = LSTM(self.units, dropout=0.2, return_sequences=True, return_state=True, name='decoder_lstm')
        # decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=[state_h, state_c])

        # # Bahdanau Attention layer
        # attn_layer = AdditiveAttention(name="additive_attention")
        # attn_out = attn_layer([encoder_outputs, decoder_outputs])
        # decoder_outputs = Concatenate(axis=-1)([decoder_outputs, attn_out])

        # dense = Dense(self.charset_size, activation='softmax')
        # decoder_softmax = TimeDistributed(dense, name="decoder_softmax")
        # decoder_outputs = decoder_softmax(decoder_outputs)

        # # The main model will turn `encoder_input_data` & `decoder_input_data`
        # # into `decoder_target_data`
        # self.model_train = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
        # self.model_train.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

        # # Define the encoder model separately.
        # self.model_encoder = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])

        # # Define the decoder model separately.
        # decoder_encoder_outputs = Input(shape=(None, self.units))
        # decoder_state_input_h = Input(shape=(self.units,))
        # decoder_state_input_c = Input(shape=(self.units,))
        # decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        # decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)

        # attn_out = attn_layer([decoder_encoder_outputs, decoder_outputs])
        # decoder_outputs = Concatenate(axis=-1)([decoder_outputs, attn_out])

        # decoder_outputs = decoder_softmax(decoder_outputs)

        # self.model_decoder = Model(inputs=[decoder_inputs, decoder_encoder_outputs, decoder_state_input_h, decoder_state_input_c],
        #                            outputs=[decoder_outputs, state_h, state_c])


        #### BACKUP CODE ####


        # # Define the main model consisting of encoder and decoder.
        # encoder_inputs = Input(shape=(None, self.charset_size), name='encoder_data')
        # encoder_lstm = Bidirectional(GRU(self.units, dropout=0.2, return_sequences=True, return_state=False), name='encoder_lstm_1')
        # encoder_outputs = encoder_lstm(encoder_inputs)

        # encoder_lstm = Bidirectional(GRU(self.units, dropout=0.2, return_sequences=False, return_state=True), name='encoder_lstm_2')
        # _, state_h, state_c = encoder_lstm(encoder_outputs)

        # # Set up the decoder, using `encoder_states` as initial state.
        # decoder_inputs = Input(shape=(None, self.charset_size), name='decoder_data')
        # # We set up our decoder to return full output sequences,
        # # and to return internal states as well. We don't use the return
        # # states in the training model, but we will use them in inference.
        # decoder_lstm = Bidirectional(GRU(self.units, dropout=0.2, return_sequences=True, return_state=True), name='decoder_lstm')
        # decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=[state_h, state_c])

        # # Bahdanau Attention layer
        # attn_layer = AdditiveAttention(name="additive_attention")
        # attn_out = attn_layer([encoder_outputs, decoder_outputs])
        # decoder_outputs = Concatenate(axis=-1)([decoder_outputs, attn_out])

        # dense = Dense(self.charset_size, activation='softmax')
        # decoder_softmax = TimeDistributed(dense, name="decoder_softmax")
        # decoder_outputs = decoder_softmax(decoder_outputs)

        # # The main model will turn `encoder_input_data` & `decoder_input_data`
        # # into `decoder_target_data`
        # self.model_train = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
        # self.model_train.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

        # # Define the encoder model separately.
        # self.model_encoder = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])

        # # Define the decoder model separately.
        # decoder_encoder_outputs = Input(shape=(None, self.units * 2))
        # decoder_state_input_h = Input(shape=(self.units,))
        # decoder_state_input_c = Input(shape=(self.units,))
        # decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        # decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)

        # attn_out = attn_layer([decoder_encoder_outputs, decoder_outputs])
        # decoder_outputs = Concatenate(axis=-1)([decoder_outputs, attn_out])

        # decoder_outputs = decoder_softmax(decoder_outputs)

        # self.model_decoder = Model(inputs=[decoder_inputs, decoder_encoder_outputs, decoder_state_input_h, decoder_state_input_c],
        #                            outputs=[decoder_outputs, state_h, state_c])


        #### BACKUP CODE ~   WORK    ####


        # Define the main model consisting of encoder and decoder.
        encoder_inputs = Input(shape=(None, self.charset_size), name='encoder_data')
        encoder_lstm = LSTM(self.units, dropout=0.2, return_sequences=True, return_state=False, name='encoder_lstm_1')
        encoder_outputs = encoder_lstm(encoder_inputs)

        encoder_lstm = LSTM(self.units, dropout=0.2, return_sequences=False, return_state=True, name='encoder_lstm_2')
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_outputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, self.charset_size), name='decoder_data')
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the return
        # states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(self.units, dropout=0.2, return_sequences=True, return_state=True, name='decoder_lstm')
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_softmax = Dense(self.charset_size, activation='softmax', name='decoder_softmax')
        decoder_outputs = decoder_softmax(decoder_outputs)

        # The main model will turn `encoder_input_data` & `decoder_input_data`
        # into `decoder_target_data`
        self.model_train = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
        self.model_train.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

        # Define the encoder model separately.
        self.model_encoder = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])

        # Define the decoder model separately.
        decoder_state_input_h = Input(shape=(self.units,))
        decoder_state_input_c = Input(shape=(self.units,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)

        decoder_outputs = decoder_softmax(decoder_outputs)
        self.model_decoder = Model(inputs=[decoder_inputs, decoder_state_input_h, decoder_state_input_c],
                                   outputs=[decoder_outputs, state_h, state_c])

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
        out = self.model_train.fit_generator(generator, steps_per_epoch, epochs=epochs, verbose=verbose,
                                             callbacks=callbacks, validation_data=validation_data,
                                             validation_steps=validation_steps, class_weight=class_weight,
                                             max_queue_size=max_queue_size, workers=workers, shuffle=shuffle,
                                             initial_epoch=initial_epoch)

        # self._set_encoder_decoder_weights()
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
        """

        self.model_encoder._make_predict_function()
        self.model_decoder._make_predict_function()
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
                x = np.array(x[0])
                batch_size, timesteps, _ = x.shape

                # Procedure for inference mode (sampling):
                # 1) Encode input and retrieve initial decoder state
                # 2) Run one step of decoder with this initial state
                #    and a start-of-sequence character as target
                #    Output will be the next target character
                # 3) Repeat with the current target character and current states

                # Encode the input as state vectors
                encoder_outputs, state_h, state_c = self.model_encoder.predict(x)

                # Create batch of empty target sequences of length 1 character
                target_sequences = np.zeros((batch_size, 1, self.charset_size))
                # Populate the first element of target sequence with the start-of-sequence character
                target_sequences[:, 0, self.charset.find(self.SOS)] = 1.0

                # Sampling loop for a batch of sequences
                # Exit condition: either hit max character limit or encounter end-of-sequence character
                decoded_tokens = [""] * batch_size

                for y in range(timesteps):
                    # `char_probs` has shape (nb_examples, 1, nb_target_chars)
                    # decoder_inputs = [target_sequences, encoder_outputs[:,y:(y + 1),:], state_h, state_c]
                    decoder_inputs = [target_sequences, state_h, state_c]
                    char_probs, h, c = self.model_decoder.predict(decoder_inputs)

                    # Reset the target sequences.
                    target_sequences = np.zeros((batch_size, 1, self.charset_size))

                    # Sample next character using argmax or multinomial mode
                    sampled_chars = []

                    for i in range(batch_size):
                        next_index = char_probs[i].argmax(axis=-1)

                        next_char = self.charset[int(next_index)]
                        decoded_tokens[i] += next_char
                        sampled_chars.append(next_char)

                        # Update target sequence with index of next character
                        target_sequences[i, 0, next_index] = 1.0

                    stop_char = set(sampled_chars)
                    if len(stop_char) == 1 and stop_char.pop() == self.EOS:
                        break

                    state_h = h
                    state_c = c

                # Sampling finished
                decoded_tokens = [" ".join(c.split()).replace(self.EOS, "") for c in decoded_tokens]
                predicts.extend(decoded_tokens)

                steps_done += 1
                if verbose == 1:
                    progbar.update(steps_done)

        finally:
            if enqueuer is not None:
                enqueuer.stop()

        return predicts

    def summary(self, output=None, target=None):
        """Show/Save model structure (summary)"""

        if target is not None:
            os.makedirs(output, exist_ok=True)

            with open(os.path.join(output, target), "w") as f:
                with redirect_stdout(f):
                    self.model_train.summary()
        self.model_train.summary()

    def load_checkpoint(self, target):
        """Restore model to construct the encoder and decoder"""

        if os.path.isfile(target):
            """Load full model (train)"""

            if self.model_train is None:
                self.compile()

            self.model_train.load_weights(target, by_name=True)
            self.model_encoder.load_weights(target, by_name=True)
            self.model_decoder.load_weights(target, by_name=True)

    def _set_encoder_decoder_weights(self):
        """Set weights into encoder and decoder models through model train layers name"""

        names = [weight.name for layer in self.model_train.layers for weight in layer.weights]
        encoder_weights, decoder_weights = [], []

        for name, weight in zip(names, self.model_train.get_weights()):
            if "encoder" in name:
                encoder_weights.append(weight)
            elif "decoder" in name:
                decoder_weights.append(weight)

        self.model_encoder.set_weights(encoder_weights)
        self.model_decoder.set_weights(decoder_weights)
