import tensorflow as tf
from encoder import Encoder
from decoder import Decoder


class PositionEmbedding(tf.keras.Model):
    def __init__(self, input_dim, output_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.position_layer = tf.keras.layers.Embedding(input_dim=5, output_dim=self.output_dim, mask_zero=True)
        self.word_layer = tf.keras.layers.Embedding(input_dim=self.input_dim, output_dim=self.output_dim,
                                                    mask_zero=True)

        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=output_dim)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization()
        self.dense_proj = tf.keras.Sequential(
            [tf.keras.layers.Dense(256, activation='relu'),
             tf.keras.layers.Dense(5)]
        )
        self.layer_norm_2 = tf.keras.layers.LayerNormalization()
        self.output_dense_1 = tf.keras.Sequential(
            [tf.keras.layers.Dense(1)]
        )
        self.output_dense_2 = tf.keras.Sequential(
            [tf.keras.layers.Dense(1)]
        )

    def call(self, inputs, training=None, mask=None):
        word, pos = inputs[:1, :], inputs[1:, :]
        embedding_output = self.word_layer(word)
        position_output = self.position_layer(pos)
        embedding_output = embedding_output + position_output
        attention_output = self.attention(embedding_output, embedding_output, attention_mask=mask)
        proj_input = self.layer_norm_1(embedding_output + attention_output)
        proj_output = self.dense_proj(proj_input)
        output = self.layer_norm_2(proj_output + proj_input)
        y = self.output_dense_1(output)
        y = tf.reshape(y, [-1])[tf.newaxis, :]
        y = self.output_dense_2(y)
        return y

    def get_config(self):
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim
        }


class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff,
                 input_vocab_size, target_vocab_size, dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               vocab_size=input_vocab_size,
                               dropout_rate=dropout_rate)

        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               vocab_size=target_vocab_size,
                               dropout_rate=dropout_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, *args, **kwargs):
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        context, x = inputs

        context = self.encoder(context)  # (batch_size, context_len, d_model)

        x = self.decoder(x, context)  # (batch_size, target_len, d_model)

        # Final linear layer output.
        logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            # b/250038731
            del logits._keras_mask
        except AttributeError:
            pass

        # Return the final output and the attention weights.
        return logits