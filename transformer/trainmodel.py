import os
import random
from utils import fileutil
import tensorflow as tf
from transformer.const import MAX_LENGTH_BOTTOM, MAX_LENGTH_TOP, BATCH_SIZE
from transformer.modelv2 import ModelV2
from keras.metrics import Precision, Recall
from transformer.f1score import F1Score


class TrainModel:
    def __init__(self, ablation_key, opt_level):
        self._key = ablation_key
        self._opt_level = opt_level

    @property
    def key(self):
        return self._key

    @property
    def opt_level(self):
        return self._opt_level

    def choose_model_train_v2(self):
        path = f'../data/step3/train_114/{self.key}/{self.opt_level}'
        if os.path.exists(path):
            self.train_model_v2(path)

    def get_dataset_v2(self, step3_dir):
        """
        modelV2版本的dataset
        """
        path = fileutil.get_files(step3_dir)
        sample_num = len(path)
        train_path = random.sample(path, k=sample_num)
        train = tf.data.TFRecordDataset(train_path)

        feature_description = {
            'top': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'bottom': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'label': tf.io.FixedLenFeature([], tf.string, default_value='')
        }

        def _parse_function(example_proto):
            dic = tf.io.parse_single_example(example_proto, feature_description)
            top = tf.io.parse_tensor(dic['top'], out_type=tf.int64)
            bottom = tf.io.parse_tensor(dic['bottom'], out_type=tf.int64)
            label = tf.io.parse_tensor(dic['label'], out_type=tf.int64)

            top = top[-MAX_LENGTH_TOP:]
            bottom = bottom[:MAX_LENGTH_BOTTOM]

            length_top = len(top)
            length_bottom = len(bottom)

            top = tf.pad(top, tf.constant([[1, 0]]) * (MAX_LENGTH_TOP - length_top))
            bottom = tf.pad(bottom, tf.constant([[0, 1]]) * (MAX_LENGTH_BOTTOM - length_bottom))

            label = tf.reshape(label, [1])
            top = tf.reshape(top, [MAX_LENGTH_TOP])
            bottom = tf.reshape(bottom, [MAX_LENGTH_BOTTOM])

            return tf.concat([top, bottom], axis=0), label

        train = train.map(_parse_function).batch(batch_size=BATCH_SIZE).prefetch(1000)
        return train

    def train_model_v2(self, data_dir):
        model = ModelV2(num_layers=1, d_model=5, num_heads=4, dff=512, vocab_size=1010)
        train = self.get_dataset_v2(data_dir)

        model.compile(optimizer='RMSprop',
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['BinaryAccuracy', Precision(name='precision'), Recall(name='recall'), F1Score()])
        csv_logger = tf.keras.callbacks.CSVLogger(f'../data/training_metrics_114_{self.key}.csv')
        # 训练
        model.fit(train, epochs=1, callbacks=csv_logger)
        # 看看模型长啥样
        model.summary()
        # 保存模型
        model_save_dir = data_dir.replace('step3', 'model2')
        os.makedirs(model_save_dir, exist_ok=True)
        model.save(model_save_dir)

    def test_model_v2(self):
        data_dir = f'../data/step3/test_100/{self.key}/{self.opt_level}'
        model = tf.keras.models.load_model(f'../data/model2/train_100/{self.key}/{self.opt_level}', custom_objects={'F1Score': F1Score})
        test = self.get_dataset_v2(data_dir)
        model.evaluate(test)


if __name__ == '__main__':
    #test_model = TrainModel(ablation_key='amno', opt_level='O0')
    #test_model.test_model_v2()
    train_model1 = TrainModel(ablation_key='ano', opt_level='O0')
    train_model1.choose_model_train_v2()
    # train_model1.test_model_v2()
    # train_model2 = TrainModel(ablation_key='amno', opt_level='O0')
    # train_model2.choose_model_train_v2()
    # train_model3 = TrainModel(ablation_key='aknos', opt_level='O0')
    # train_model3.choose_model_train_v2()
    # train_model4 = TrainModel(ablation_key='abinor', opt_level='O0')
    # train_model4.choose_model_train_v2()
