import gin
import logging
from absl import app, flags
import tensorflow as tf
from train import Trainer
from input_pipeline.datasets import load
from utilss import utils_params, utils_misc
from models.model import lstm, gru
from evaluation.metrics import confusionmatrix
from evaluation.eval import Eval
from input_pipeline.tfrecords import TFRecords

FLAGS = flags.FLAGS
train = False
flags.DEFINE_boolean('train', train, 'Specify whether to train or test a model.')

FLAG1 = flags.FLAGS
TFrecord_needed = False
flags.DEFINE_boolean('TFrecord_needed', TFrecord_needed, 'Specify if TFRecord is ready.')

@gin.configurable
def main(argv):
    # change the number to decide which model.
    Choose_model = ['lstm', 'gru']
    model_flag = Choose_model[0]

    # change the number to decide the operation.
    Choose = ['evaluate', 'confusionmatrix', 'visu.plot_visu']
    test_flag = Choose[0]

    # generate folder structures
    run_paths = utils_params.gen_run_folder()
    print(run_paths)
    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    # utils_params.save_config(run_paths['path_gin'], gin.config_str())

    if FLAG1.TFrecord_needed:
        # write TFRecord
        data_dir = " /home/data/HAPT_dataset/RawData/"
        records_dir = "/home/data/human_activity_recognition/data_tfrecords/"
        TFRecords().generate_tfrecords(data_dir=data_dir, records_dir=records_dir)

    else:
        pass

    # setup pipeline
    ds_train, ds_val, ds_test = load()

    if model_flag == 'lstm':
        model = lstm()
    elif model_flag == 'gru':
        model = gru()

    model.summary()

    # Train
    if FLAGS.train:
        trainer = Trainer(model, ds_train, ds_val, run_paths)
        for _ in trainer.train():
            continue

    # Evaluation
    else:
        # Load checkpoints
        if model_flag == 'lstm':
            checkpoint = tf.train.Checkpoint(step=tf.Variable(0), model=model)
            manager = tf.train.CheckpointManager(checkpoint,
                                                 directory="/Users/mengxiangyuan/Desktop/DL_Lab/dl-lab-22w-team08"
                                                           "/experiments/run_2023-01-07_lstm_ACC_94.4/ckpts/",
                                                 max_to_keep=3)
            checkpoint.restore(manager.latest_checkpoint)
            if manager.latest_checkpoint:
                tf.print("restore")
            else:
                tf.print("bad")

        elif model_flag == 'gru':
            checkpoint = tf.train.Checkpoint(step=tf.Variable(0), model=model)
            manager = tf.train.CheckpointManager(checkpoint,
                                                 directory="/Users/mengxiangyuan/Desktop/DL_Lab/dl-lab-22w-team08"
                                                           "/experiments/run_2023-01-07_gru_Acc_95.8/ckpts/",
                                                 max_to_keep=3)
            checkpoint.restore(manager.latest_checkpoint)
            if manager.latest_checkpoint:
                tf.print("restore")
            else:
                tf.print("bad")

        # Evaluation
        evaluation = Eval(model, ds_test, run_paths)
        if test_flag == 'evaluate':
            evaluation.evaluate()
        elif test_flag == 'confusionmatrix':
            confusionmatrix(model, ds_test)
        elif test_flag == 'visu.plot_visu':
            evaluation.plot_visu(data_dir="/home/data/HAPT_dataset/RawData/")


if __name__ == "__main__":
    # run_path = "/Users/mengxiangyuan/Desktop/DL_Lab/dl-lab-22w-team08/"
    app.run(main)

