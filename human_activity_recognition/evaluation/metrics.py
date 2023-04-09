import os
import logging
import numpy as np
from matplotlib import pyplot as plt


class DrawConfusionMatrix:

    def __init__(self, labels_name, normalize=True):
        """normalize：if set number to percentage"""
        self.normalize = normalize
        self.labels_name = labels_name
        self.num_classes = len(labels_name)
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype="float32")
        self.mat = np.zeros((self.num_classes, self.num_classes), dtype="float32")

    def update(self, labels, predicts):
        """write labels predicts as one dime vector"""
        for label, predict in zip(labels, predicts):
            self.matrix[label, predict] += 1

        return self.matrix

    def getMatrix(self, normalize=True):
        """
        if normalize=True，percentage，
        if normalize=False，number
        Returns a matrix with number or percentage
        """
        if normalize:
            per_sum = self.matrix.sum(axis=1)  # row-sum
            for i in range(self.num_classes):
                self.matrix[i] = (self.matrix[i] / per_sum[i])  # 百分比转换
            self.matrix = np.around(self.matrix, 2)  # 保留2位小数点
            self.matrix[np.isnan(self.matrix)] = 0  # 可能存在NaN，将其设为0
        return self.matrix

    def drawMatrix(self):
        self.matrix = self.getMatrix(self.normalize)

        plt.figure(figsize=(15, 15))

        plt.imshow(self.matrix, cmap=plt.cm.Blues)  # 仅画出颜色格子，没有值

        plt.title("Normalized confusion matrix")  # title

        plt.xlabel("Predict label", fontsize=12)
        plt.ylabel("Truth label", fontsize=12)

        plt.yticks(range(self.num_classes), self.labels_name, rotation=45)  # y轴标签
        plt.xticks(range(self.num_classes), self.labels_name, rotation=45)  # x轴标签

        for x in range(self.num_classes):
            for y in range(self.num_classes):
                value = float(format('%.2f' % self.matrix[y, x]))  # 数值处理
                plt.text(x, y, value, verticalalignment='center', horizontalalignment='center',
                         color='darkgrey')  # 写值

        # plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域

        plt.colorbar()  # 色条

        # lstm
        path = '/Users/mengxiangyuan/Desktop/DL_Lab/dl-lab-22w-team08/experiments/run_2023-01-07_lstm_ACC_94.4/plot/'

        # gru
        # path = '/Users/mengxiangyuan/Desktop/DL_Lab/dl-lab-22w-team08/experiments/run_2023-01-07_gru_Acc_95.8/plot/'

        plot_path = os.path.join(path, "cm_1.png")
        plt.savefig(plot_path)


labels_name = ['walking', 'walking_upstairs', 'walking_downstairs', 'sitting', 'standing', 'laying', 'stand-sit',
               'sit-stand', 'sit-lie', 'lie-sit', 'stand-lie', 'lie-stand']

drawconfusionmatrix = DrawConfusionMatrix(labels_name=labels_name)  # 实例化


def confusionmatrix(model, ds_test):

    logging.info(f'\n==================== Starting draw Confusionmatrix ====================')

    for features, labels in ds_test:
        prediction = model(features, training=False)
        predict_np = np.argmax(prediction.numpy(), axis=1)
        labels_np = labels.numpy()
        drawconfusionmatrix.update(labels_np, predict_np)

    drawconfusionmatrix.drawMatrix()
    confusion_mat = drawconfusionmatrix.getMatrix()
    print(confusion_mat)

    logging.info(f'\n==================== Finished draw Confusionmatrix ====================')
