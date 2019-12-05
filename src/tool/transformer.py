"""
Implementation of Transformer Model using TensorFlow 2.0.

References:
    Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and
    Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin.
    "Attention Is All You Need", 2017
    arXiv, URL: https://arxiv.org/abs/1706.03762
"""

import os
import logging
import tensorflow as tf

from contextlib import redirect_stdout
from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Progbar, GeneratorEnqueuer

from tensorflow.keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
from tensorflow.keras.layers import Lambda, Embedding

tf.get_logger().setLevel(logging.ERROR)


class Transformer():
    """
    Transformer Model.

    References:
        Bryan M. Li and FOR.ai
        A Transformer Chatbot Tutorial with TensorFlow 2.0, 2019
        Medium: https://medium.com/tensorflow/a-transformer-chatbot-tutorial-with-tensorflow-2-0-88bf59e66fe2

        Tensorflow documentation
        Transformer model for language understanding
        URL: https://www.tensorflow.org/tutorials/text/transformer

        Trung Tran
        Create The Transformer With Tensorflow 2.0
        Machine Talk: https://machinetalk.org/2019/04/29/create-the-transformer-with-tensorflow-2-0/
        Github: https://github.com/ChunML/NLP/tree/master/machine_translation
        Jupyter Notebook: https://colab.research.google.com/drive/1YhN8ZCZhrv18Hw0a_yIkuZ5tTh4EZDuG#scrollTo=ha0dNJogUPQN
    """

    def __init__(self, tokenizer, num_layers, units, d_model, num_heads, dropout=0.0):
        self.tokenizer = tokenizer
        self.num_layers = num_layers
        self.units = units
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout

        global MAXLENGH
        MAXLENGH = self.tokenizer.maxlen

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
        """Restore model to construct transformer/encoder/decoder"""

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
                patience=10,
                verbose=verbose)
        ]

        return callbacks

    def compile(self, learning_rate=None):
        """Build models (train, encoder and decoder)"""

        inputs = Input(shape=(None,), name="inputs")
        dec_inputs = Input(shape=(None,), name="dec_inputs")

        self.encoder = self._encoder_model(vocab_size=self.tokenizer.vocab_size,
                                           num_layers=self.num_layers,
                                           units=self.units,
                                           d_model=self.d_model,
                                           num_heads=self.num_heads,
                                           dropout=self.dropout)

        enc_outputs, enc_padding_mask = self.encoder(inputs=inputs)

        self.decoder = self._decoder_model(vocab_size=self.tokenizer.vocab_size,
                                           num_layers=self.num_layers,
                                           units=self.units,
                                           d_model=self.d_model,
                                           num_heads=self.num_heads,
                                           dropout=self.dropout)

        dec_outputs = self.decoder(inputs=[dec_inputs, enc_outputs, enc_padding_mask])

        if learning_rate is None:
            learning_rate = CustomSchedule(d_model=self.d_model)
            self.learning_schedule = True
        else:
            self.learning_schedule = False

        optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        self.model = Model(inputs=[inputs, dec_inputs], outputs=dec_outputs, name="transformer")
        self.model.compile(optimizer=optimizer, loss=self.loss_func, metrics=[self.accuracy])

    def _encoder_model(self, vocab_size, num_layers, units, d_model, num_heads, dropout, name="encoder"):
        """Build encoder model with your layers"""

        inputs = Input(shape=(None,), name="inputs")
        enc_padding_mask = Lambda(self.create_padding_mask, output_shape=(1, 1, None), name="enc_padding_mask")(inputs)

        embeddings = Embedding(vocab_size, d_model)(inputs)
        embeddings *= tf.math.sqrt(tf.cast(d_model, dtype="float32"))
        embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

        outputs = Dropout(rate=dropout)(embeddings)

        for i in range(num_layers):
            outputs = self._encoder_layer(units=units,
                                          d_model=d_model,
                                          num_heads=num_heads,
                                          dropout=dropout,
                                          name=f"encoder_layer_{i}",
                                          )([outputs, enc_padding_mask])

        return Model(inputs=inputs, outputs=[outputs, enc_padding_mask], name=name)

    def _encoder_layer(self, units, d_model, num_heads, dropout, name="encoder_layer"):
        """Build encoder block layers"""

        inputs = Input(shape=(None, d_model), name="inputs")
        padding_mask = Input(shape=(1, 1, None), name="padding_mask")

        attention = MultiHeadAttention(
            d_model, num_heads, name="attention")({
                "query": inputs,
                "key": inputs,
                "value": inputs,
                "mask": padding_mask
            })

        attention = Dropout(rate=dropout)(attention)
        attention = LayerNormalization(epsilon=1e-6)(inputs + attention)

        outputs = Dense(units=units, activation="relu")(attention)
        outputs = Dense(units=d_model)(outputs)
        outputs = Dropout(rate=dropout)(outputs)
        outputs = LayerNormalization(epsilon=1e-6)(attention + outputs)

        return Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)

    def _decoder_model(self, vocab_size, num_layers, units, d_model, num_heads, dropout, name="decoder"):
        """Build decoder model with your layers"""

        inputs = Input(shape=(None,), name="inputs")
        enc_outputs = Input(shape=(None, d_model), name="encoder_outputs")
        dec_padding_mask = Input(shape=(1, 1, None), name="dec_padding_mask")

        look_ahead_mask = Lambda(self.create_look_ahead_mask, output_shape=(1, None, None), name="look_ahead_mask")(inputs)

        embeddings = Embedding(vocab_size, d_model)(inputs)
        embeddings *= tf.math.sqrt(tf.cast(d_model, dtype="float32"))
        embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

        outputs = Dropout(rate=dropout)(embeddings)

        for i in range(num_layers):
            outputs = self._decoder_layer(units=units,
                                          d_model=d_model,
                                          num_heads=num_heads,
                                          dropout=dropout,
                                          name=f"decoder_layer_{i}",
                                          )(inputs=[outputs, enc_outputs, look_ahead_mask, dec_padding_mask])

        outputs = Dense(units=vocab_size, name="outputs")(outputs)

        return Model(inputs=[inputs, enc_outputs, dec_padding_mask], outputs=outputs, name=name)

    def _decoder_layer(self, units, d_model, num_heads, dropout, name="decoder_layer"):
        """Build decoder block layers"""

        inputs = Input(shape=(None, d_model), name="inputs")
        enc_outputs = Input(shape=(None, d_model), name="encoder_outputs")
        look_ahead_mask = Input(shape=(1, None, None), name="look_ahead_mask")
        padding_mask = Input(shape=(1, 1, None), name="padding_mask")

        attention1 = MultiHeadAttention(
            d_model, num_heads, name="attention_1")(inputs={
                "query": inputs,
                "key": inputs,
                "value": inputs,
                "mask": look_ahead_mask
            })
        attention1 = LayerNormalization(epsilon=1e-6)(attention1 + inputs)

        attention2 = MultiHeadAttention(
            d_model, num_heads, name="attention_2")(inputs={
                "query": attention1,
                "key": enc_outputs,
                "value": enc_outputs,
                "mask": padding_mask
            })
        attention2 = Dropout(rate=dropout)(attention2)
        attention2 = LayerNormalization(epsilon=1e-6)(attention2 + attention1)

        outputs = Dense(units=units, activation="relu")(attention2)
        outputs = Dense(units=d_model)(outputs)
        outputs = Dropout(rate=dropout)(outputs)
        outputs = LayerNormalization(epsilon=1e-6)(outputs + attention2)

        return Model(inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask], outputs=outputs, name=name)

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

        # remove ReduceLROnPlateau (if exist) when use schedule learning rate
        if callbacks and self.learning_schedule:
            callbacks = [x for x in callbacks if not isinstance(x, ReduceLROnPlateau)]

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

                for sentence in x:
                    en_input = tf.expand_dims(sentence, axis=0)
                    en_output, en_mask = self.encoder.predict(en_input)

                    dec_input = tf.expand_dims([self.tokenizer.SOS], axis=0)

                    for _ in range(self.tokenizer.maxlen):
                        dec_output = self.decoder.predict([dec_input, en_output, en_mask])
                        predicted_id = tf.cast(tf.argmax(dec_output[:, -1:, :], axis=-1), dtype="int32")

                        if tf.equal(predicted_id, self.tokenizer.EOS):
                            break

                        # concatenated the predicted_id to the output
                        # which is given to the decoder as its input.
                        dec_input = tf.concat((dec_input, predicted_id), axis=-1)

                    dec_input = tf.squeeze(dec_input, axis=0)
                    dec_input = self.tokenizer.decode([i for i in dec_input[1:] if i < self.tokenizer.vocab_size])
                    predicts.extend(" ".join(dec_input.split()))

                # Sampling finished
                predicts = [self.tokenizer.remove_tokens(x) for x in predicts]

                steps_done += 1
                if verbose == 1:
                    progbar.update(steps_done)

        finally:
            enqueuer.stop()

        return predicts

    @staticmethod
    def loss_func(y_true, y_pred):
        """Loss function with SparseCategoryCrossentropy and mask to filter out padded tokens"""

        y_true = tf.reshape(y_true, shape=(-1, MAXLENGH))
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")(y_true, y_pred)
        mask = tf.cast(tf.not_equal(y_true, 0), dtype="float32")
        loss = tf.multiply(loss, mask)
        return tf.reduce_mean(loss)

    @staticmethod
    def accuracy(y_true, y_pred):
        """Accuracy function with SparseCategoryCrossentropy and mask to filter out padded tokens"""

        y_true = tf.reshape(y_true, shape=(-1, MAXLENGH))
        accuracy = tf.metrics.SparseCategoricalAccuracy()(y_true, y_pred)
        return accuracy

    @staticmethod
    def create_look_ahead_mask(x):
        """Mask the future tokens for decoder inputs at the 1st attention block"""

        seq_len = tf.shape(x)[1]
        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        padding_mask = Transformer.create_padding_mask(x)
        return tf.maximum(look_ahead_mask, padding_mask)

    @staticmethod
    def create_padding_mask(x):
        """Mask the encoder outputs for the 2nd attention block"""

        mask = tf.cast(tf.math.equal(x, 0), dtype="float32")
        return mask[:, tf.newaxis, tf.newaxis, :]


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-Head Attention (self attention), make the core of Transformer Model.
    The Scaled Dot-Product Attention is basically similar to Luong attention (dot score function).
    """

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = Dense(units=d_model)
        self.key_dense = Dense(units=d_model)
        self.value_dense = Dense(units=d_model)

        self.dense = Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # scaled dot-product attention
        # scaled_attention = scaled_dot_product_attention(query, key, value, mask)
        matmul_qk = tf.matmul(query, key, transpose_b=True)

        # scale matmul_qk
        depth = tf.cast(tf.shape(key)[-1], dtype="float32")
        logits = matmul_qk / tf.math.sqrt(depth)

        # add the mask to zero out padding tokens
        if mask is not None:
            logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k)
        attention_weights = tf.nn.softmax(logits, axis=-1)
        scaled_attention = tf.matmul(attention_weights, value)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # concatenation of heads
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # final linear layer
        outputs = self.dense(concat_attention)

        return outputs


"""
Positional Encoding and Custom Schedule.
Inject some information about the relative or absolute position of the tokens in the sequence.

References:
    Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and
    Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin.
    "Attention Is All You Need", 2017
    arXiv, URL: https://arxiv.org/abs/1706.03762

    Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, Yann N. Dauphin
    "Convolutional Sequence to Sequence Learning", 2017
    arXiv, URL: https://arxiv.org/abs/1705.03122

    Sho Takase, Naoaki Okazaki.
    "Positional Encoding to Control Output Sequence Length", 2019
    arXiv, URL: https://arxiv.org/abs/1904.07418
"""


class PositionalEncoding(tf.keras.layers.Layer):
    """Positional encoding (features)"""

    def __init__(self, position, d_model, name="PositionalEncodingLayer"):
        super(PositionalEncoding, self).__init__(name=name)
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, dtype="float32"))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype="float32")[:, tf.newaxis],
            i=tf.range(d_model, dtype="float32")[tf.newaxis, :],
            d_model=d_model)
        # apply sin to even index in the array
        sines = tf.math.sin(angle_rads[:, 0::2])
        # apply cos to odd index in the array
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, dtype="float32")

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Custom schedule of the learning rate with warmup_steps.
    From original paper "Attention is all you need".
    """

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, dtype="float32")

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
