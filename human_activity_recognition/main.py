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


parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--model', choices= ['lstm', 'gru'], default = 'lstm' help='choose model')
parser.add_argument('--mode', choices=['train','test'], default = 'train', help='train or test')
parser.add_argument('--evaluation', choices=['evaluate', 'confusionmatrix', 'visu.plot_visu'], default = 'evaluate', help='evaluation methods')
parser.add_argument('--checkpoint-file', type=str, default='./ckpts/',
                    help='Path to checkpoint.')

args = parser.parse_args()



@gin.configurable
def main(argv):
   

    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    ds_train, ds_val, ds_test = load()

    if args.model == 'lstm':
        model = lstm()
    elif args.model == 'gru':
        model = gru()
    else:
        print('Error, no model is fund')

    model.summary()

    # Train
    if args.mode == 'train':
        trainer = Trainer(model, ds_train, ds_val, run_paths)
        for _ in trainer.train():
            continue

    # Evaluation
    else:
         checkpoint = tf.train.Checkpoint(step=tf.Variable(0), model=model)
            manager = tf.train.CheckpointManager(checkpoint,directory = args.checkpoint_file, max_to_keep=3)
            checkpoint.restore(manager.latest_checkpoint)
            if manager.latest_checkpoint:
                tf.print("restore")
            else:
                tf.print("Error")
        # Evaluation
        evaluation = Eval(model, ds_test, run_paths)
        if args.evaluation == 'evaluate':
            evaluation.evaluate()
        elif args.evaluation == 'confusionmatrix':
            confusionmatrix(model, ds_test)
        elif args.evaluation == 'visu.plot_visu':
            evaluation.plot_visu(data_dir="/home/data/HAPT_dataset/RawData/")


if __name__ == "__main__":
    app.run(main)

