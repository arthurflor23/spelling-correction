"""
Implementation of Transformer Model using TensorFlow 2.0.

References:
    Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and
    Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin.
    "Attention Is All You Need", 2017
    arXiv, URL: https://arxiv.org/abs/1706.03762
"""

import os
import numpy as np
import tensorflow as tf

from contextlib import redirect_stdout
from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Progbar, GeneratorEnqueuer
from tensorflow.keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


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

            self.model.load_weights(target)

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

    def compile(self, learning_rate=None, initial_step=0):
        """Build models (train, encoder and decoder)"""

        enc_input = Input(shape=(None,), name="enc_input")
        dec_input = Input(shape=(None,), name="dec_input")
        enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(enc_input, dec_input)

        self.encoder = Encoder(num_layers=self.num_layers,
                               d_model=self.d_model,
                               num_heads=self.num_heads,
                               dff=self.units,
                               input_vocab_size=self.tokenizer.vocab_size,
                               maximum_position_encoding=self.tokenizer.vocab_size,
                               rate=self.dropout)

        self.decoder = Decoder(num_layers=self.num_layers,
                               d_model=self.d_model,
                               num_heads=self.num_heads,
                               dff=self.units,
                               target_vocab_size=self.tokenizer.vocab_size,
                               maximum_position_encoding=self.tokenizer.vocab_size,
                               rate=self.dropout)

        enc_output = self.encoder(enc_input, enc_padding_mask)
        dec_output, _ = self.decoder(dec_input, enc_output, look_ahead_mask, dec_padding_mask)

        if learning_rate is None:
            learning_rate = CustomSchedule(d_model=self.d_model, initial_step=initial_step)
            self.learning_schedule = True
        else:
            self.learning_schedule = False

        optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        self.model = Model(inputs=[enc_input, dec_input], outputs=dec_output, name="transformer")
        self.model.compile(optimizer=optimizer, loss=loss_func, metrics=["accuracy"])

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
                    enc_input = tf.expand_dims(sentence, axis=0)
                    dec_input = tf.expand_dims([self.tokenizer.SOS], axis=0)

                    for _ in range(self.tokenizer.maxlen):
                        enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(enc_input, dec_input)

                        enc_output = self.encoder(enc_input, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
                        dec_output, _ = self.decoder(dec_input, enc_output, look_ahead_mask, dec_padding_mask)

                        # select the last word from the seq_len dimension
                        predictions = dec_output[:, -1:, :]  # (batch_size, 1, vocab_size)
                        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), dtype=tf.int32)

                        # return the result if the predicted_id is equal to the end token
                        if tf.equal(predicted_id, self.tokenizer.EOS):
                            break

                        # concatentate the predicted_id to the output which is given to the decoder as its input.
                        dec_input = tf.concat([dec_input, predicted_id], axis=-1)

                    dec_input = tf.squeeze(dec_input, axis=0)
                    dec_input = self.tokenizer.decode(dec_input)

                    predicts.append(self.tokenizer.remove_tokens(dec_input))

                steps_done += 1
                if verbose == 1:
                    progbar.update(steps_done)

        finally:
            enqueuer.stop()

        return predicts


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model, name="enc_embedding")
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate, f"enc_layer_{i}") for i in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate, name="enc_dropout")

    def call(self, x, mask=None):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)

        return x  # (batch_size, input_seq_len, d_model)

    def get_config(self):
        """Return the config of the layer"""

        config = super(Encoder, self).get_config()
        return config


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, name="enc_layer"):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads, name=f"{name}_attention")
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm_1")
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm_2")

        self.dropout1 = tf.keras.layers.Dropout(rate, name=f"{name}_dropout_1")
        self.dropout2 = tf.keras.layers.Dropout(rate, name=f"{name}_dropout_2")

    def call(self, x, mask=None):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

    def get_config(self):
        """Return the config of the layer"""

        config = super(EncoderLayer, self).get_config()
        return config


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model, name="dec_embedding")
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate, name=f"dec_layer_{i}") for i in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate, name="dec_dropout")

        self.dec_output = tf.keras.layers.Dense(target_vocab_size, name="dec_dense")

    def call(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, look_ahead_mask, padding_mask)

            attention_weights["decoder_layer{}_block1".format(i + 1)] = block1
            attention_weights["decoder_layer{}_block2".format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        output = self.dec_output(x)

        return output, attention_weights

    def get_config(self):
        """Return the config of the layer"""

        config = super(Decoder, self).get_config()
        return config


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, name="dec_layer"):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads, name=f"{name}_attention_1")
        self.mha2 = MultiHeadAttention(d_model, num_heads, name=f"{name}_attention_2")

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm_1")
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm_2")
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm_3")

        self.dropout1 = tf.keras.layers.Dropout(rate, name=f"{name}_dropout_1")
        self.dropout2 = tf.keras.layers.Dropout(rate, name=f"{name}_dropout_2")
        self.dropout3 = tf.keras.layers.Dropout(rate, name=f"{name}_dropout_3")

    def call(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2

    def get_config(self):
        """Return the config of the layer"""

        config = super(DecoderLayer, self).get_config()
        return config


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

    def get_config(self):
        """Return the config of the layer"""

        config = super(MultiHeadAttention, self).get_config()
        return config


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation="relu"),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, initial_step=0, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.initial_step = initial_step
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step + self.initial_step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def create_masks(inp=None, tar=None):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def loss_func(y_true, y_pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    loss_ = loss_object(y_true, y_pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)
