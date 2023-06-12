import tensorflow as tf
from transformer.encoder import EncoderLayer
from transformer.positionalembedding import PositionalEmbedding
from transformer.attention import SelfAttention, CrossAttention
from transformer.feedforward import FeedForward


class ModelV2Part1(EncoderLayer):
    """
    第一部分，和EncoderLayer一模一样
    """
    pass


class ModelV2Part2(tf.keras.Model):
    """
    第二部分，与DecoderLayer的差别主要在于。1.只输入输出一次 2.不是使用的CausalSelfAttention，而是SelfAttention
    """
    def __init__(self,
                 *,
                 d_model,
                 num_heads,
                 dff,
                 dropout_rate=0):
        super(ModelV2Part2, self).__init__()

        self.self_attention = SelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, inputs, *args, **kwargs):
        # 这一层的输入应该是来源于两个部分，第一部分输入，是直接给到的输入，第二部分，是ModelV2Part1的输出作为输入
        # 输入为 ((batch, MAX_LENGTH, embed的维度), (batch, MAX_LENGTH, embed的维度))
        x, context = inputs
        x = self.self_attention(x=x)
        x = self.cross_attention(inputs=(x, context))

        # FeedForward
        x = self.ffn(x)
        return x


class ModelV2(tf.keras.Model):
    """
    model_v2,这个model由5个部分组成
    1.embedding层A
    2.encoder层，即ModelV2Part1
    3.embedding层B
    4.第二个encoder层，即ModelV2Part2
    5.全联接层，输出0/1
    """

    def __init__(self,
                 *,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 vocab_size,
                 dropout_rate=0):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # 第1部分，embedding
        self.pos_embedding1 = PositionalEmbedding(
            vocab_size=vocab_size, output_dim=d_model, pos_dim=21)

        # 第2部分，encoder
        self.enc_layers1 = [
            ModelV2Part1(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)]

        # 第3部分，第二个embedding层
        self.pos_embedding2 = PositionalEmbedding(
            vocab_size=vocab_size, output_dim=d_model, pos_dim=21)

        # 第4部分，另一个encoder层
        self.enc_layers2 = [
            ModelV2Part2(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)]

        # dropout层暂时未用上，为以后的实验准备
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        # 第5部分，全联接层输出结果
        # 5.1 这时候输出的还是一个shape为 (batch, MAX_LENGTH, embed_dim) 的张量，通过这一步变为 (batch, MAX_LENGTH, 1)
        # self.flat_layer = tf.keras.layers.Dense(1)
        # 5.2 再把它变为 (batch, 1)的结果，去和label比较
        self.final_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x, *args, **kwargs):
        # x的shape为 ((batch, sequence_length), (batch, sequence_length)),两个tensor组成的元组
        # x的shape为(batch, 2 * sequence_length)
        # 1.左边部分的处理
        # top和top_pos做嵌入
        length = x.shape[1]
        top, bottom = x[:, 0:length // 2], x[:, length // 2:]
        print('top shape = ', top.shape)  # top shape = (None, 120)
        print('bottom shape = ', bottom.shape)  # bottom shape = (None, 120)

        left = self.pos_embedding1(top)
        print('top_embedding shape = ', left.shape)         # top_embedding shape = (None, 30, 10)

        # 所有的encoder layer层依次调用，形状不变
        for i in range(self.num_layers):
            left = self.enc_layers1[i](left)
        print('top_trans shape = ', left.shape)             # top_trans shape = (None, 30, 10)

        # 2.右边部分的处理
        # bottom和bottom_pos做嵌入
        right = self.pos_embedding2(bottom)
        print('bottom_embedding shape = ', right.shape)     # bottom_embedding shape = (None, 30, 10)

        # 所有的encoder layer层依次调用，形状不变
        for i in range(self.num_layers):
            # 右边部分的输入是embed和左边的输出
            right = self.enc_layers2[i](inputs=(right, left))
        print('all_trans shape = ', right.shape)            # all_trans shape = (None, 30, 10)

        # 把输出变为0/1
        result = tf.reshape(right, [-1, right.shape[1] * right.shape[2]])
        print('reshape = ', result.shape)                   # reshape = (None, 300)
        result = self.final_layer(result)
        print('result shape = ', result.shape)              # result shape = (None, 1)
        return result

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_layers': self.num_layers,
            'd_model': self.d_model
        })
        return config
