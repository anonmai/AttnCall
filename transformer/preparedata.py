import tensorflow as tf
from utils import fileutil
import os
import multiprocessing
from transformer import tokengeneration


class PrepareData:
    def __init__(self, prepare_dirs, optlevel):
        self._prepare_dirs = prepare_dirs
        self.optlevel = optlevel

    @property
    def prepare_dirs(self):
        return self._prepare_dirs

    def get_dataset(self, path, layer):
        def generator():
            with open(path, mode='r') as f:
                line = f.readline()
                i = 0
                data = ['', '', '']
                while line:
                    i += 1
                    line = line.rstrip()
                    data[(i-1) % 3] = line
                    line = f.readline()
                    if i % 3 == 0:
                        yield data[0], data[1], data[2]

        def vectorization(top: tf.Tensor, bottom: tf.Tensor, label: tf.Tensor):
            def word_vectorization(word: tf.Tensor):
                split_tensor = tf.strings.split(word)
                word_tensor = layer.call(split_tensor)
                word_tensor = tf.squeeze(word_tensor, axis=1)
                return word_tensor

            top_tensor = word_vectorization(top)
            bottom_tensor = word_vectorization(bottom)
            label_tensor = tf.strings.to_number(label, out_type='int64')
            return top_tensor, bottom_tensor, label_tensor

        dataset = tf.data.Dataset.from_generator(generator=generator,
                                                 output_signature=(
                                                     tf.TensorSpec(shape=(), dtype=tf.string),
                                                     tf.TensorSpec(shape=(), dtype=tf.string),
                                                     tf.TensorSpec(shape=(), dtype=tf.string),
                                                 ))
        return dataset.map(vectorization)

    def feature_value(self, value):
        st = tf.io.serialize_tensor(value)
        if isinstance(st, type(tf.constant(0))):
            st = st.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[st]))

    def serialize_example(self, top, bottom, label):
        feature = {
            "top": self.feature_value(top),
            "bottom": self.feature_value(bottom),
            "label": self.feature_value(label)
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def write_dataset(self, old_path, new_path, key):
        def generator():
            for feature in dataset:
                result = self.serialize_example(*feature)
                yield result

        tokens = tokengeneration.legal_token_list(key, self.optlevel)
        layer = tf.keras.layers.TextVectorization(vocabulary=tokens)
        dataset = self.get_dataset(old_path, layer)
        serialized_features_dataset = tf.data.Dataset.from_generator(
            generator, output_types=tf.string, output_shapes=())
        with tf.io.TFRecordWriter(new_path) as writer:
            for e in serialized_features_dataset:
                writer.write(e.numpy())
        print('write dataset', new_path)

    def step3(self):
        pool = multiprocessing.Pool(2)
        for d in self.prepare_dirs:
            key = os.path.split(d)[1]
            files = fileutil.get_files(d)
            # 开一个进程池并行写入，充分利用cpu
            for file in files:
                path = file.replace('step2', 'step3')
                path_dir = os.path.split(path)
                os.makedirs(path_dir[0], exist_ok=True)
                pool.apply_async(func=self.write_dataset, args=(file, path, key))
        pool.close()
        pool.join()


if __name__ == '__main__':
    dirs = [
        '../data/step2/train_114/ano',
        # '../data/step2/test_100/amno',
        # '../data/step2/test_100/abinor',
        # '../data/step2/test_100/aknos',
    ]
    optlevel = 'O0'
    prepare = PrepareData(dirs, optlevel)
    prepare.step3()
