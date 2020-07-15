# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 16:41:20 2020

@author: acfba
"""

image_size = 299


from mittens import Mittens
from mittens import GloVe
import tensorflow as tf

from sklearn.feature_extraction.text import CountVectorizer


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_captions(tfrecord):
    record_iterator = tf.python_io.tf_record_iterator(tfrecord)
    annotations = []
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        annotations.append(example.features.feature['labels'].bytes_list.value[0])
    return annotations


def preProBuildWordVocab(sentence_iterator, word_count_threshold=20):  # function from Andre Karpathy's NeuralTalk

    vectorizer = CountVectorizer(lowercase=False, dtype=np.float64, ngram_range=(1, 1),
                                 stop_words=('Lesions', 'START', 'END'))

    X = vectorizer.fit_transform(sentence_iterator)

    Xc = (X.T * X)

    vocab = vectorizer.get_feature_names()

    vocab2 = [w.encode() for w in vocab]

    return Xc.todense(), vocab, vocab2, np.diagonal(Xc.todense())


glove_filename = "/home/catarinabarata/glove.6B.50d.txt"


def glove2dict(glove_filename):
    with open(glove_filename, encoding="utf8") as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
        embed = {line[0]: np.array(list(map(float, line[1:])))
                 for line in reader}
    return embed


def loadGloVe(co_occur, dim_embed, vocab, is_fine):
    if is_fine:
        glove_original = glove2dict(glove_filename)

        mittens_model = Mittens(n=dim_embed, max_iter=5000)

        embeddings = mittens_model.fit(
            np.asarray(co_occur),
            vocab=vocab,
            initial_embedding_dict=glove_original)
    else:
        glove_model = GloVe(n=dim_embed, max_iter=5000)
        embeddings = glove_model.fit(np.asarray(co_occur))

    return embeddings

def parse_fn_train(example):
    "Parse TFExample records"
    example_fmt = {
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature((), tf.string),
        "labels": tf.FixedLenFeature((), tf.string, "")
    }

    parsed = tf.parse_single_example(example, example_fmt)

    height = tf.cast(parsed['height'], tf.int32)
    width = tf.cast(parsed['width'], tf.int32)

    image = tf.decode_raw(parsed['image_raw'], tf.uint8)

    image = tf.reshape(image, (height, width, 3))

    return image, parsed["labels"]

def read_and_decode(dataset, batch_size, is_training, data_size):
    if is_training:
        dataset = dataset.shuffle(buffer_size=data_size, reshuffle_each_iteration=True)
        dataset = dataset.prefetch(buffer_size=data_size // batch_size)
        dataset = dataset.map(map_func=parse_fn_train, num_parallel_calls=tf.data.experimental.AUTOTUNE,
                              )
        dataset = dataset.batch(batch_size, drop_remainder=False)

    else:
        dataset = dataset.prefetch(buffer_size=data_size // batch_size)
        dataset = dataset.map(map_func=parse_fn_train, num_parallel_calls=tf.data.experimental.AUTOTUNE,
                              )
        dataset = dataset.batch(batch_size, drop_remainder=False)

    return dataset

def transform_data(img, is_training):
    image_dense = tf.map_fn(
        lambda x: inception_preprocessing.preprocess_image(x, image_size, image_size, is_training=is_training, crop=is_training,
                                                           add_image_summaries=False, fast_mode=False), img,
        dtype=tf.float32)

    image_dense = tf.to_float(tf.image.convert_image_dtype(image_dense, tf.uint8))

    image_aux = image_dense

    image_dense = densenet_preprocessing._SCALE_FACTOR * tf.map_fn(
        lambda x: densenet_preprocessing._mean_image_subtraction(x, [densenet_preprocessing._R_MEAN,
                                                                     densenet_preprocessing._G_MEAN,
                                                                     densenet_preprocessing._B_MEAN]), image_dense,
        dtype=tf.float32)

    return image_dense