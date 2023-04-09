import gin
import keras
import logging
from keras import layers


@gin.configurable
def SimpleRNN(
        n_classes,
        window_length,
        rnn_units,
        rnn_num,
        rnn_dropout,
        dense_units,
        dense_num,
        dense_dropout,
        kernel_initializer,
        return_sequence=False,):

    """
    Specify the structure of the RNN model

    Args:
        n_classes (int): number of output classes, 12
        window_length (int): length of the sliding window, 250
        rnn_units (int): number of rnn units
        rnn_num (int): number of recurrent layers
        rnn_dropout (float): dropout rate between rnn layers
        dense_units (int): number of dense units
        dense_num (int): number of dense layers
        dense_dropout (float): dropout rate between dense layers
        kernel_initializer (str): Kernel initialization method
        return_sequence (bool): Use False here when using the S2T method

    Returns:
        model
    """

    model = keras.Sequential([keras.Input(shape=(window_length, 6)), ])

    # rnn layer
    for _ in range(rnn_num):
        layer = layers.SimpleRNN(units=rnn_units, return_sequences=True,
                                 dropout=rnn_dropout, kernel_initializer=kernel_initializer)
        model.add(layer)
        model.add(layers.MaxPool1D(2))
        model.add(layers.BatchNormalization())
    layer = layers.SimpleRNN(units=rnn_units, return_sequences=return_sequence,
                             dropout=rnn_dropout, kernel_initializer=kernel_initializer)
    model.add(layer)

    # dense layer
    for _ in range(dense_num):
        model.add(layers.Dense(dense_units, kernel_regularizer=keras.regularizers.L1L2(l1=0.01, l2=0.01)))
        model.add(layers.Dropout(dense_dropout))

    model.add(layers.Dense(n_classes, activation="softmax"))

    logging.info(f"rnn input shape:  {model.input_shape}")
    logging.info(f"rnn output shape: {model.output_shape}")

    return model


@gin.configurable
def lstm(
        n_classes,
        window_length,
        rnn_units,
        rnn_num,
        rnn_dropout,
        dense_units,
        dense_num,
        dense_dropout,
        kernel_initializer,
        return_sequence=False,):

    """
    Specify the structure of the RNN model

    Args:
        n_classes (int): number of output classes, 12
        window_length (int): length of the sliding window, 250
        rnn_units (int): number of rnn units
        rnn_num (int): number of recurrent layers
        rnn_dropout (float): dropout rate between rnn layers
        dense_units (int): number of dense units
        dense_num (int): number of dense layers
        dense_dropout (float): dropout rate between dense layers
        kernel_initializer (str): Kernel initialization method
        return_sequence (bool): Use False here when using the S2T method

    Returns:
        model
    """

    model = keras.Sequential([keras.Input(shape=(window_length, 6)), ])

    # rnn layer
    for _ in range(rnn_num):
        layer = layers.LSTM(units=rnn_units, return_sequences=True,
                            dropout=rnn_dropout, kernel_initializer=kernel_initializer)
        model.add(layer)
        model.add(layers.MaxPool1D(2))
        model.add(layers.BatchNormalization())
    layer = layers.LSTM(units=rnn_units, return_sequences=return_sequence,
                        dropout=rnn_dropout, kernel_initializer=kernel_initializer)
    model.add(layer)

    # dense layer
    for _ in range(dense_num):
        model.add(layers.Dense(dense_units, kernel_regularizer=keras.regularizers.L1L2(l1=0.01, l2=0.01)))
        model.add(layers.Dropout(dense_dropout))

    model.add(layers.Dense(n_classes, activation="softmax"))

    logging.info(f"rnn input shape:  {model.input_shape}")
    logging.info(f"rnn output shape: {model.output_shape}")

    return model


@gin.configurable
def gru(
        n_classes,
        window_length,
        rnn_units,
        rnn_num,
        rnn_dropout,
        dense_units,
        dense_num,
        dense_dropout,
        kernel_initializer,
        return_sequence=False,):

    """
    Specify the structure of the RNN model

    Args:
        n_classes (int): number of output classes, 12
        window_length (int): length of the sliding window, 250
        rnn_units (int): number of rnn units
        rnn_num (int): number of recurrent layers
        rnn_dropout (float): dropout rate between rnn layers
        dense_units (int): number of dense units
        dense_num (int): number of dense layers
        dense_dropout (float): dropout rate between dense layers
        kernel_initializer (str): Kernel initialization method
        return_sequence (bool): Use False here when using the S2T method

    Returns:
        model
    """

    model = keras.Sequential([keras.Input(shape=(window_length, 6)), ])

    # rnn layer
    for _ in range(rnn_num):
        layer = layers.GRU(units=rnn_units, return_sequences=True,
                           dropout=rnn_dropout, kernel_initializer=kernel_initializer)
        model.add(layer)
        model.add(layers.MaxPool1D(2))
        model.add(layers.BatchNormalization())
    layer = layers.GRU(units=rnn_units, return_sequences=return_sequence,
                       dropout=rnn_dropout, kernel_initializer=kernel_initializer)
    model.add(layer)

    # dense layer
    for _ in range(dense_num):
        model.add(layers.Dense(dense_units, kernel_regularizer=keras.regularizers.L1L2(l1=0.01, l2=0.01)))
        model.add(layers.Dropout(dense_dropout))

    model.add(layers.Dense(n_classes, activation="softmax"))

    logging.info(f"rnn input shape:  {model.input_shape}")
    logging.info(f"rnn output shape: {model.output_shape}")

    return model


if __name__ == '__main__':
    0
