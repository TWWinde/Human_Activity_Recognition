import gin
import sys
import logging
import tensorflow as tf
from input_pipeline.preprocessing import preprocess


@gin.configurable
def load(name, data_dir, batch_size):
    if name == "hapt":
        logging.info(f"Preparing dataset {name}...")
        raw_train_ds = tf.data.TFRecordDataset(data_dir + "/train.tfrecords")
        raw_val_ds = tf.data.TFRecordDataset(data_dir + "/validation.tfrecords")
        raw_test_ds = tf.data.TFRecordDataset(data_dir + "/test.tfrecords")

        # decode raw data
        decoded_ds_train = raw_train_ds.map(prepare_record)
        decoded_ds_val = raw_val_ds.map(prepare_record)
        decoded_ds_test = raw_test_ds.map(prepare_record)

        # to have a look
        # groundtruth = []
        # for data, label in enumerate(decoded_ds_train):
        #    groundtruth.append(np.asarray(label))
        # groundtruth = np.asarray(groundtruth)
        # print(groundtruth[0:252])

        return prepare(decoded_ds_train, decoded_ds_val, decoded_ds_test, batch_size=batch_size)

    else:
        logging.error('The HAR data set is an optional part! Currently only the HAPT dataset is available!')
        sys.exit(0)


@gin.configurable
def prepare(ds_train, ds_val, ds_test, batch_size, caching):
    # Prepare training dataset

    ds_train = ds_train.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if caching:
        ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(64)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare validation dataset
    ds_val = ds_val.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.batch(batch_size)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare test dataset
    ds_test = ds_test.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val, ds_test


def prepare_record(record):
    """Parse and decode tfrecords file"""

    name_to_features = {
        'features': tf.io.FixedLenFeature([], tf.string),
        'labels': tf.io.FixedLenFeature([], tf.string),
    }
    record = tf.io.parse_single_example(record, name_to_features)
    features = tf.io.parse_tensor(record['features'], out_type=tf.double)
    labels = tf.io.parse_tensor(record['labels'], out_type=tf.double)
    record = (features, labels)

    return record
