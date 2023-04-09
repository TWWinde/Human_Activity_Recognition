import os
import gin
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy.stats import zscore, mode


@gin.configurable
class Eval:

    def __init__(self, exp_user, model, ds_test, run_paths):

        self.model = model
        self.run_paths = run_paths
        self.ds_test = ds_test
        self.n_classes = model.output_shape[1]
        self.exp_user = exp_user
        self.window_length = 250
        self.color_name = {1: 'salmon', 2: 'navajowhite', 3: 'lemonchiffon', 4: 'palegreen',
                           5: 'mediumspringgreen', 6: 'paleturquoise', 7: 'lightskyblue', 8: 'cornflowerblue',
                           9: 'mediumslateblue', 10: 'mediumorchid', 11: 'violet', 12: 'lightpink',
                           0: 'white'}
        self.labels_name = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING',
                            'STAND_TO_SIT', 'SIT_TO_STAND', 'SIT_TO_LIE', 'LIE_TO_SIT', 'STAND_TO_LIE', 'LIE_TO_STAND']

    def evaluate(self):
        """ Test and count test set accuracy """

        logging.info(f'\n==================== Starting Evaluation ====================')

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        eval_loss = tf.keras.metrics.Mean(name='eval_loss')
        eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='eval_accuracy')

        for images, labels in self.ds_test:
            # Show comparison of forecast and ground truth
            label = labels.numpy()
            print('ground-truth', label.flatten())

            predictions = self.model(images, training=False)
            print('prediction.', np.argmax(predictions.numpy(), axis=-1))
            print('..............................................................')

            t_loss = loss_object(labels, predictions)
            eval_loss(t_loss)
            eval_accuracy(labels, predictions)

        template = ' Test Loss: {}, Test SparseCategoricalAcc: {}'
        logging.info(template.format(eval_loss.result(), eval_accuracy.result() * 100, ))

        logging.info(f'\n==================== Finished Evaluation ====================')

        return

    def plot_all_visu(self, data_dir):
        """
        Visualize all experimental samples at once

        Args:
            data_dir (str): The path to the directory where the data is stored
        """
        logging.info(f'\n==================== Starting all data Visualization ====================')

        for user in range(1, 30):
            if user <= 9:
                for experiment in range((user - 1) * 2, (user - 1) * 2 + 1):
                    self.plot_visu(self, data_dir)
            if user == 10:
                for experiment in range((user - 1) * 2, (user - 1) * 2 + 2):
                    self.plot_visu(self, data_dir)
            else:
                for experiment in range(user * 2, user * 2 + 1):
                    self.plot_visu(self, data_dir)

        logging.info(f'\n==================== Finished all data Visualization ====================')

    def plot_visu(self, data_dir):
        """
        Visualize experimental samples

        Args:
            data_dir (str): The path to the directory where the data is stored
        """
        logging.info(f'\n==================== Starting Visualization ====================')

        # load data
        labels = pd.read_csv(os.path.join(data_dir, "labels.txt"), sep=" ", header=None)
        acc_data = pd.read_csv(
            os.path.join(data_dir, f"acc_exp{str(self.exp_user[0]).zfill(2)}_user{str(self.exp_user[1]).zfill(2)}.txt"),
            sep=" ", header=None)
        gyro_data = pd.read_csv(
            os.path.join(data_dir, f"gyro_exp{str(self.exp_user[0]).zfill(2)}_user{str(self.exp_user[1]).zfill(2)}.txt"),
            sep=" ", header=None)

        sensor_data = pd.concat([acc_data, gyro_data], axis=1)  # 横向表拼接 行对齐
        sensor_data.columns = ["acc_1", "acc_2", "acc_3", "gyro_1", "gyro_2", "gyro_3"]

        # normalization, Initialization
        sensor_data_norm = zscore(sensor_data, axis=0)
        file_length = sensor_data.shape[0]
        groundtruth_color_values = []
        prediction_color_values = []
        sensor_data_norm['label'] = 0

        # get data
        for index, (exp, user, act, sco, eco) in labels.iterrows():
            if exp == self.exp_user[0] and user == self.exp_user[1]:
                sensor_data_norm.loc[sco:eco, 'label'] = act

        # Obtain gt and predicted labels for sequences (batched into specified window lengths)
        for i in range(0, file_length, self.window_length):

            # get label for sequence
            gt_seq_labels = sensor_data_norm.loc[i:i + self.window_length - 1, 'label'].to_numpy()
            groundtruth_color_values.append(gt_seq_labels)

            if mode(gt_seq_labels)[0][0] == 0:
                prediction_color_values.append(gt_seq_labels)

            else:
                features = sensor_data_norm.values[i:i + self.window_length, :-1]
                features = np.expand_dims(features, 0)
                predictions = self.model(features, training=False)
                predicted_label = np.argmax(predictions) + 1
                predicted_labels = np.full(self.window_length, predicted_label)
                prediction_color_values.append(predicted_labels)

        prediction_color_values = np.concatenate(prediction_color_values).ravel()
        groundtruth_color_values = np.concatenate(groundtruth_color_values).ravel()

        # Plot Visualization diagram
        self.plot_file(values=groundtruth_color_values, title="Accelerations_Ground-Truth",
                       x=sensor_data_norm['acc_1'].values,
                       y=sensor_data_norm['acc_2'].values,
                       z=sensor_data_norm['acc_3'].values,
                       legend_x='acc_X', legend_y='acc_Y', legend_z='acc_Z', savename="acc_gt")

        self.plot_file(values=prediction_color_values, title="Accelerations_Predictions",
                       x=sensor_data_norm['acc_1'].values,
                       y=sensor_data_norm['acc_2'].values,
                       z=sensor_data_norm['acc_3'].values,
                       legend_x='acc_X', legend_y='acc_Y', legend_z='acc_Z', savename="acc_pd")

        self.plot_file(values=groundtruth_color_values, title="Gyroscope_Ground-Truth",
                       x=sensor_data_norm['gyro_1'].values,
                       y=sensor_data_norm['gyro_2'].values,
                       z=sensor_data_norm['gyro_3'].values,
                       legend_x='gyro_X', legend_y='gyro_Y', legend_z='gyro_Z', savename="gyro_gt")

        self.plot_file(values=prediction_color_values, title="Gyroscope_Predictions",
                       x=sensor_data_norm['gyro_1'].values,
                       y=sensor_data_norm['gyro_2'].values,
                       z=sensor_data_norm['gyro_3'].values,
                       legend_x='gyro_X', legend_y='gyro_Y', legend_z='gyro_Z', savename="gyro_pd")

        # self.plot_colormap()

        logging.info(f'\n==================== Finished Visualization ====================')

    def plot_file(self, title, values, x, y, z, legend_x, legend_y, legend_z, savename):
        """ Draw visual graphs based on relevant parameters """

        plt.figure(figsize=(20, 4))
        for index, color in enumerate(values):
            plt.axvspan(index, index + 1, facecolor=self.color_name[color], alpha=0.6)
        plt.plot(x, color='orangered', label=legend_x)
        plt.plot(y, color='royalblue', label=legend_y)
        plt.plot(z, color='seagreen', label=legend_z)
        plt.title(title)
        plt.legend(loc="lower right")

        # lstm
        path = '/Users/mengxiangyuan/Desktop/DL_Lab/dl-lab-22w-team08/experiments/run_2023-01-07_lstm_ACC_94.4/plot/visu/'

        # gru
        # path = '/Users/mengxiangyuan/Desktop/DL_Lab/dl-lab-22w-team08/experiments/run_2023-01-07_gru_Acc_95.8/plot/visu/'

        plot_path = os.path.join(path, f"E{str(self.exp_user[0]).zfill(2)}_U{str(self.exp_user[1]).zfill(2)}_"
                                 + savename + "_visu.png")
        plt.savefig(plot_path)
        logging.info(f'Saving "{title}" plot to: {plot_path}')

    def plot_colormap(self):
        """ draw color map """

        plt.figure(figsize=(25, 5))
        plt.title('Color Map')
        x = np.arange(0, 12, 1)
        plt.bar(x, height=1, width=1, align='center', color=list(self.color_name.values()))
        plt.yticks([])
        plt.xticks(x, self.labels_name, rotation=60)
        plt.margins(0)
        ax = plt.gca()
        ax.spines['right'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_color('black')
        fig = plt.gcf()
        fig.set_size_inches(20, 2)
        plt.tight_layout()
        path = '/Users/mengxiangyuan/Desktop/DL_Lab/dl-lab-22w-team08/human_activity_recognition/plot/'
        plot_path = os.path.join(path, 'colormap.png')
        plt.savefig(plot_path)
        logging.info(f'Saving colormap to: {plot_path}')

