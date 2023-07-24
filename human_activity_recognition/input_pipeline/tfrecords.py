import os
import gin
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.stats import zscore, mode


def histogram(feature, label):
    """Visualize the data distribution"""

    _, train_label = serialize(feature, label)
    labels = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING',
              'STAND_TO_SIT', 'SIT_TO_STAND', 'SIT_TO_LIE', 'LIE_TO_SIT', 'STAND_TO_LIE', 'LIE_TO_STAND']
    # hist = plt.hist(train_label, bins='auto')
    plt.title("Class distribution")
    plt.xlabel("Activities")
    plt.xticks(np.arange(1, 13), labels=labels, fontsize=8, rotation=45)
    plt.ylabel("Numbers")
    plt.tight_layout()
    plt.show()

    return


def oversampling_data(labels, features):
    # oversampling 

    labels_oversampled = np.empty((0, labels.shape[1]))
    features_oversampled = np.empty((0, features.shape[1], features.shape[2]))  # shape (0,1,2) 

    activities, activity_counts = np.unique(labels, return_counts=True)

    max_activity = np.max(activity_counts[1:7])
    max_transition_activity = np.max(activity_counts[7:])

    # remove zero
    for activity in activities:
        activity_indices = np.where(labels == activity)[0]
        if activity < 7:
            indices = np.random.choice(activity_indices, size=max_activity, replace=True)
        else:
            indices = np.random.choice(activity_indices, size=max_transition_activity, replace=True)

        labels_oversampled = np.append(labels_oversampled, labels[indices], axis=0)
        features_oversampled = np.append(features_oversampled, features[indices], axis=0)

    return labels_oversampled, features_oversampled


def serialize(labels, features):
    # Create a tf.train.Example, ready to write to the file

    labels_of_bytes = tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(labels).numpy()]))
    feature_of_bytes = tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(features).numpy()]))
    example_proto = tf.train.Example(
        features=tf.train.Features(feature={"features": feature_of_bytes, "labels": labels_of_bytes}))
    example = example_proto.SerializeToString()

    return example


@gin.configurable
class TFRecords:

    def __init__(self, window_length, window_shift):

        self.window_length = window_length
        self.window_shift = window_shift

    def define_window(self, data_format):
        # S2L, delete unavailable data
        # Creating the basic window format 

        labels_list = []
        features_list = []

        # delete first 5s and last 5s data. 
        for index in range(self.window_length, len(data_format) - self.window_length, self.window_shift):
            window_data = data_format.iloc[index: (index + self.window_length)].values

            label, count = mode(window_data[:, 6]).mode[0], mode(window_data[:, 6]).count[0]
            if label >= 7: 
                if (count / self.window_length) <= 0.45:
                    label = 0
            else:
                if (count / self.window_length) <= 0.85:
                    label = 0

            labels_list.append(label)
            features_list.append(window_data[:, :-1])

        labels_list = np.expand_dims(np.array(labels_list), axis=1)
        features_list = np.array(features_list)

        return labels_list, features_list

    def generate_tfrecords(self, data_dir, records_dir):
        """
        After pre-processing all data and generating TFRecords files

        Args:
            data_dir (str): The path to the directory where the data is stored
            records_dir: The path to the directory where the TFRecords data is stored
        """

        features_train = np.empty(shape=(0, self.window_length, 6))
        features_val = np.empty(shape=(0, self.window_length, 6))
        features_test = np.empty(shape=(0, self.window_length, 6))

        labels_train = np.empty(shape=(0, 1))
        labels_val = np.empty(shape=(0, 1))
        labels_test = np.empty(shape=(0, 1))

        exp_now, user_now = -1, -1
        labels = pd.read_csv(os.path.join(data_dir, "labels.txt"), sep=" ", header=None)

        for index, (exp, user, act, sco, eco) in labels.iterrows():

            if (exp != exp_now) or (user != user_now):
                exp_now, user_now = exp, user

                # read files data to dataframe
                acc_data = pd.read_csv(
                    os.path.join(data_dir, f"acc_exp{str(exp_now).zfill(2)}_user{str(user_now).zfill(2)}.txt"),
                    sep=" ", header=None)
                gyro_data = pd.read_csv(
                    os.path.join(data_dir, f"gyro_exp{str(exp_now).zfill(2)}_user{str(user_now).zfill(2)}.txt"),
                    sep=" ", header=None)
                sensor_data = pd.concat([acc_data, gyro_data], axis=1) 
                sensor_data.columns = ["acc_1", "acc_2", "acc_3", "gyro_1", "gyro_2", "gyro_3"]

                # Initialization
                sensor_data_norm = zscore(sensor_data, axis=0)
                sensor_data_norm["label"] = 0

            sensor_data_norm.loc[sco:eco, "label"] = act
            win_labels, win_features = self.define_window(data_format=sensor_data_norm)

            # Divide the data set as required
            if user_now in range(1, 22):
                labels_train = np.append(labels_train, win_labels, axis=0)
                features_train = np.append(features_train, win_features, axis=0)
            elif user_now in range(28, 31):
                labels_val = np.append(labels_val, win_labels, axis=0)
                features_val = np.append(features_val, win_features, axis=0)
            elif user_now in range(22, 28):
                labels_test = np.append(labels_test, win_labels, axis=0)
                features_test = np.append(features_test, win_features, axis=0)

        # Tagging samples with no activity label, label = 0
        blank_activities_train = np.where(labels_train == 0)[0]
        blank_activities_val = np.where(labels_val == 0)[0]
        blank_activities_test = np.where(labels_test == 0)[0]

        # delete samples (labels and ) that do not have an activity label, label=0
        labels_train = np.delete(labels_train, blank_activities_train, axis=0)
        labels_val = np.delete(labels_val, blank_activities_val, axis=0)
        labels_test = np.delete(labels_test, blank_activities_test, axis=0)

        features_train = np.delete(features_train, blank_activities_train, axis=0)
        features_val = np.delete(features_val, blank_activities_val, axis=0)
        features_test = np.delete(features_test, blank_activities_test, axis=0)

        # Oversampling
        labels_train, features_train = oversampling_data(labels_train, features_train)

        # shuffle dataset
        labels_train, features_train = shuffle(labels_train, features_train)

        '''
        # number of samples
        logging.info("")
        logging.info(f"training samples:    {features_train.shape[0]}")
        logging.info(f"validation samples:  {features_val.shape[0]}")
        logging.info(f"test samples:        {features_test.shape[0]}")
        logging.info("")
        '''

        # create the files for TFRecords'
        # os.makedirs(records_dir)

        # Write data into TFRecords files
        with tf.io.TFRecordWriter(records_dir + "/train.tfrecords") as writer:
            for label, feature in tf.data.Dataset.from_tensor_slices((labels_train, features_train)):
                example = serialize(label, feature)
                writer.write(example)

        with tf.io.TFRecordWriter(records_dir + "/validation.tfrecords") as writer:
            for label, feature in tf.data.Dataset.from_tensor_slices((labels_val, features_val)):
                example = serialize(label, feature)
                writer.write(example)

        with tf.io.TFRecordWriter(records_dir + "/test.tfrecords") as writer:
            for label, feature in tf.data.Dataset.from_tensor_slices((labels_test, features_test)):
                example = serialize(label, feature)
                writer.write(example)

        return True



TFRecords().generate_tfrecords(data_dir=data_dir, records_dir=records_dir)
