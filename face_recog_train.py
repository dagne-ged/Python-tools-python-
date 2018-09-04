# coding=utf-8

import numpy as np
import tensorflow as tf
import os
import os.path
from PIL import Image
from squeeze_net import SqueezeNet


DIR = r'F:\数据集\PIE\PIE - 副本'
CLASS = 68  # 68个人
LEARNING_RATE = 1e-4


def pics_to_arrs(directory: str='.', to_npy=False, from_npy=False):
    """
    这个数据集中同一个人的脸都放在一个文件夹中，‘s1’-‘s68’
    :param directory:包含‘s1’-‘s68’文件夹的父文件夹
    :param to_npy: 保存到npy文件'pic_arrs.npy'、'pic_labs.npy'中
    :param from_npy:直接从npy文件中读取数组
    :return: pic_arrs, pic_labs, 即X与Y
    """
    if from_npy:
        pic_arrs = np.load('pic_arrs.npy')
        pic_labs = np.load('pic_labs.npy')
        return pic_arrs, pic_labs
    pic_arr_li = []
    pic_lab_li = []
    for person in os.listdir(directory):
        # 注意person是相对路径,即名字
        if not person.startswith('s'):
            continue    # 跳过不是的文件
        person_num = int(person[1:]) - 1  # 编号、y标签, 从0开始
        person_dir = os.path.join(directory, person)
        for file in os.listdir(person_dir):
            if not file.endswith(('jpg', 'jpeg')):
                continue    # 跳过不是的文件
            pic_abs = os.path.join(person_dir, file)
            ima = np.array(Image.open(pic_abs), dtype=np.float32)
            ima = ima[:, :, np.newaxis]     # cnn一定要3维的图像,channel last
            pic_arr_li.append(ima)      # 得到X
            pic_lab_li.append(person_num)   # 得到y

    print(ima.shape)
    length = len(pic_lab_li)
    print('length is ', length, type(pic_lab_li[0]))
    rand_mask = np.array(range(length))
    np.random.shuffle(rand_mask)
    pic_arrs = np.array(pic_arr_li)
    pic_arrs = pic_arrs[rand_mask]
    pic_labs = np.array(pic_lab_li)
    print(pic_labs[0:10])
    pic_labs = pic_labs[rand_mask]  # 打乱顺序
    print(pic_labs[0:10])
    if to_npy:
        np.save('pic_arrs.npy', pic_arrs)
        np.save('pic_labs.npy', pic_labs)

    return pic_arrs, pic_labs


def train_input_fn(features, labels, batch_size=32, buffer=1000):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(buffer_size=buffer).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    # features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


def face_model_fn(features, labels, mode, params):
    model = SqueezeNet(CLASS, (64, 64, 1))
    imas = features
    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = model(imas, training=False)
        predictions = {
            'classes': tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits),
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions
        )
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

        logits = model(imas, training=True)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        # sparse_softmax:labels must be an index in [0, num_classes)
        accuracy = tf.metrics.accuracy(
            labels=labels, predictions=tf.argmax(logits, axis=1))

        # Name tensors to be logged with LoggingTensorHook.
        tf.identity(LEARNING_RATE, 'learning_rate')
        tf.identity(loss, 'cross_entropy')
        tf.identity(accuracy[1], name='train_accuracy')

        # Save accuracy scalar to Tensorboard output.
        tf.summary.scalar('train_accuracy', accuracy[1])

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))
    if mode == tf.estimator.ModeKeys.EVAL:
        logits = model(imas, training=False)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops={
                'accuracy':
                    tf.metrics.accuracy(
                        labels=labels, predictions=tf.argmax(logits, axis=1)),
            })


def train_exp(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    pic_arrs, pic_labs = pics_to_arrs(from_npy=True)  # X，Y

    ckpt_config = tf.estimator.RunConfig(save_checkpoints_steps=10)
    classifier = tf.estimator.Estimator(
        model_fn=face_model_fn,
        model_dir='./modelCkpt',     # 模型检查点路径
        config=ckpt_config
    )
    classifier.train(input_fn=
        lambda: train_input_fn(pic_arrs, pic_labs, batch_size=4, buffer=20),
        steps=10
    )
    print('train over')
    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda: eval_input_fn(pic_arrs, pic_labs, 4))
    test_arrs = pic_arrs[100:108]
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
    predictions = classifier.predict(
        input_fn=lambda: eval_input_fn(test_arrs, None, 4)   # predict的时候labels为None
    )
    for p in predictions:
        cls = p['classes']
        prob = p['probabilities'][cls]
        print('class is ', cls, ' prob is ', prob)


TRAIN_SIZE = 1400
def train_process(_):
    """
    真正进行训练的函数
    :param _: tf.app.run会传一个参数给这里，用不到
    :return:
    """
    tf.logging.set_verbosity(tf.logging.INFO)
    pic_arrs, pic_labs = pics_to_arrs(from_npy=True)  # X，Y
    train_arrs = pic_arrs[:TRAIN_SIZE]
    train_labs = pic_labs[:TRAIN_SIZE]  # 训练数据,生成数据时已经洗牌过了
    test_arrs = pic_arrs[TRAIN_SIZE:]
    test_labs = pic_labs[TRAIN_SIZE:]   # 测试数据
    ckpt_config = tf.estimator.RunConfig(save_checkpoints_steps=200)
        # 200步存一次检查点
    classifier = tf.estimator.Estimator(
        model_fn=face_model_fn,
        model_dir='./modelCkpt',     # 模型检查点路径
        config=ckpt_config
    )
    classifier.train(input_fn=
        lambda: train_input_fn(pic_arrs, pic_labs, batch_size=32, buffer=2000),
        max_steps=4000
    )   # batch是32. shuffle的buffer是2000因为总共有1632个数据，最大的训练步骤是4000步
    print('---train over---')
    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda: eval_input_fn(pic_arrs, pic_labs, 4))
    print('Test set accuracy: {accuracy:0.3f}'.format(**eval_result))


def draft():
    aa = list(range(4))
    print(aa)
    li = [0,2,3,1]
    bb = aa[li]
    print(bb)


if __name__ == '__main__':
    # pics_to_arrs(DIR, to_npy=True)
    # tf.app.run(train_exp)
    # draft()