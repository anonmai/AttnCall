import tensorflow as tf


class BaseAttention(tf.keras.layers.Layer):
    """
    基类
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
    """
    q为decoder，k，v为encoder
    """

    def call(self, inputs, *args, **kwargs):
        x, context = inputs
        attn_output = self.mha(
            query=x,
            key=context,
            value=context)

        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class SelfAttention(BaseAttention):
    """
    q，k，v均为自己
    """

    def call(self, x, *args, **kwargs):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class CausalSelfAttention(BaseAttention):
    """
    mask attention
    """

    def call(self, x, *args, **kwargs):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask=True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x
