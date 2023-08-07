import io
import keras
import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers
from asm2vec import Asm2Vec, Asm2VecLSTM
import random

'''
本模块利用word2vec或者lstm，对预处理后的汇编代码token进行embedding学习
'''

# 训练数据所在目录
token2vec_path = '../data/token_ins'
# Token学习embedding的维度
EMBEDDING_DIM = 30
# 负采样随机数种子
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
# 这个百分比指的是，在指令中，长度为 1，2，3的指令的长度的百分比，根据观察得到
PERCENT = [0.1, 0.3, 0.6]
# 词汇量，根据对已有数据观察得到
VOCAB_SIZE = 128
# 训练超参
BATCH_SIZE = 1024
BUFFER_SIZE = 10000


def generate_lstm_data(sequences):
    """
    获得lstm词嵌入方式的训练数据
    :param sequences: 序列集
    :return: target的shape为(sample_size, 4), labels的shape为(sample_size, )
    """
    count = len(sequences)
    target, labels = [], []
    # 获得正采样
    for sequence in tqdm.tqdm(sequences):
        target.append(sequence)
        labels.append(1)

    # 获得负采样，不同指令长度的数量和实际的百分比相同
    percent_num = np.array(PERCENT)
    percent_num = (percent_num * count).astype(int)
    for i in range(len(percent_num)):
        for _ in range(percent_num[i]):
            words = []
            for _ in range(i + 1):
                words.append(random.randint(1, VOCAB_SIZE - 1))
            for _ in range(2 - i):
                words.append(0)
            words = np.array(words)
            target.append(words)
            labels.append(0)

    return target, labels


def generate_w2v_data(sequences, window_size, num_ns):
    """
    获得word2vec的训练数据
    :param sequences: 序列集
    :param window_size: 滑动窗口长度
    :param num_ns: 负采样数量
    :return:
    """
    # Elements of each training example are appended to these lists.
    targets, contexts, labels = [], [], []

    # Build the sampling table for `vocab_size` tokens.
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(VOCAB_SIZE)

    # Iterate over all sequences (sentences) in the dataset.
    for sequence in tqdm.tqdm(sequences):

        # Generate positive skip-gram pairs for a sequence (sentence).
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequence,
            vocabulary_size=VOCAB_SIZE,
            sampling_table=sampling_table,
            window_size=window_size,
            negative_samples=0)

        # Iterate over each positive skip-gram pair to produce training examples
        # with a positive context word and negative samples.
        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(
                tf.constant([context_word], dtype="int64"), 1)
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,
                num_true=1,
                num_sampled=num_ns,
                unique=True,
                range_max=VOCAB_SIZE,
                seed=SEED,
                name="negative_sampling")

            # Build context and label vectors (for one target word)
            negative_sampling_candidates = tf.expand_dims(
                negative_sampling_candidates, 1)

            context = tf.concat([context_class, negative_sampling_candidates], 0)
            label = tf.constant([1] + [0] * num_ns, dtype="int64")

            # Append each element from the training example to global lists.
            targets.append(target_word)
            contexts.append(context)
            labels.append(label)

    return targets, contexts, labels


def training_data(model, file_name):
    """
    训练数据
    :param model: 需要使用的模型,传入一个产生模型的函数
    :param file_name: 文件前缀名，用于区分不同模型产生的文件
    :return:
    """
    text_ds = tf.data.TextLineDataset(token2vec_path).filter(lambda x: tf.cast(tf.strings.length(x), bool))
    sequence_length = 3
    vectorize_layer = keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode='int',
        output_sequence_length=sequence_length)
    vectorize_layer.adapt(text_ds.batch(1024))
    text_vector_ds = text_ds.batch(1024).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()
    sequences = list(text_vector_ds.as_numpy_iterator())

    asm2vec = model(sequences)

    weights = asm2vec.get_layer('embed_layer').get_weights()[0]
    vocab = vectorize_layer.get_vocabulary()
    output_file(weights, vocab, file_name)


def _lstm_model(sequences):
    """
    lstm方式词嵌入模型
    :param sequences: 序列数据集
    :return: 模型
    """
    target, label = generate_lstm_data(sequences)
    target1 = np.array(target)
    label1 = np.array(label)
    dataset = tf.data.Dataset.from_tensor_slices((target1, label1))
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
    asm2vec = Asm2VecLSTM(VOCAB_SIZE, EMBEDDING_DIM)
    asm2vec.compile(optimizer='adam',
                    loss=tf.keras.losses.BinaryCrossentropy(),
                    metrics=['accuracy'])
    asm2vec.fit(dataset, epochs=20)

    return asm2vec


def _word2vec_model(sequences):
    """
    word2vec方式词嵌入模型
    :param sequences: 序列数据集
    :return: 模型
    """
    targets, contexts, labels = generate_w2v_data(
        sequences=sequences,
        window_size=3,
        num_ns=2)
    targets = np.array(targets)
    contexts = np.array(contexts)[:, :, 0]
    labels = np.array(labels)
    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
    asm2vec = Asm2Vec(VOCAB_SIZE, EMBEDDING_DIM)
    asm2vec.compile(optimizer='adam',
                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
    asm2vec.fit(dataset, epochs=20)
    return asm2vec


def output_file(weights, vocab, file_name):
    """
    输出文件
    :param weights: 这是词嵌入结果
    :param vocab: 词汇
    :param file_name: 文件名，用于区分不同词嵌入方式
    :return:
    """
    out_v = io.open('./data/'+file_name+'vectors.tsv', 'w', encoding='utf-8')
    out_m = io.open('./data/'+file_name+'metadata.tsv', 'w', encoding='utf-8')
    for index, word in enumerate(vocab):
        if index == 0:
            continue  # skip 0, it's padding.
        vec = weights[index]
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        out_m.write(word + "\n")
    out_v.close()
    out_m.close()


if __name__ == '__main__':
    training_data(_lstm_model, 'lstm')
    training_data(_word2vec_model, 'w2v')
