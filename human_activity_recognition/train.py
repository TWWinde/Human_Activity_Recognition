import datetime
import os
import gin
import tensorflow as tf
import logging


@gin.configurable
class Trainer(object):
    def __init__(self, model, ds_train, ds_val, run_paths,
                 total_steps, log_interval, ckpt_interval, acc, loss_weight=1, acc_weight=1):

        logging.info(f'All relevant data from {run_paths["path_model_id"]}')

        # Summary Writer

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        current_dir = os.path.dirname(__file__)
        tensorboard_log_dir = os.path.join(current_dir, 'logs')
        log_dir = os.path.join(tensorboard_log_dir, current_time)
        logging.info(f"Tensorboard output will be stored in: {log_dir}")
        self.train_log_dir = os.path.join(log_dir, 'train')
        self.val_log_dir = os.path.join(log_dir, 'validation')

        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.val_summary_writer = tf.summary.create_file_writer(self.val_log_dir)

        # Loss objective
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        # self.optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.001)
        lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.001,
                                                                 decay_steps=1000,
                                                                 alpha=0.1)
        self.optimizer = tf.keras.optimizers.Adam(lr_scheduler)

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval
        self.loss_weight = loss_weight
        self.acc_weight = acc_weight
        self.acc = acc

        # Checkpoint Manager
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0), model=self.model, optimizer=self.optimizer)
        self.manager = tf.train.CheckpointManager(self.checkpoint, directory=run_paths["path_ckpts_train"],
                                                  max_to_keep=3)
        logging.info(f"All checkpoints will be stored in: {run_paths['path_ckpts_train']}")

        # ...

    @tf.function
    def train_step(self, features, labels):
        loss_weight_vector = tf.squeeze(tf.where(labels > 5, self.loss_weight, 1))
        acc_weight_vector = tf.squeeze(tf.where(labels > 5, self.acc_weight, 1))
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(features, training=True)
            loss = self.loss_object(labels, predictions, sample_weight=loss_weight_vector)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss(loss)
        self.train_accuracy.update_state(labels, predictions, sample_weight=acc_weight_vector)

    @tf.function
    def val_step(self, features, labels):
        loss_weight_vector = tf.squeeze(tf.where(labels > 5, self.loss_weight, 1))
        acc_weight_vector = tf.squeeze(tf.where(labels > 5, self.acc_weight, 1))
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(features, training=False)
        t_loss = self.loss_object(labels, predictions, sample_weight=loss_weight_vector)
        self.val_loss(t_loss)
        self.val_accuracy.update_state(labels, predictions, sample_weight=acc_weight_vector)

    def write_scalar_summary(self, step):
        """ Write scalar summary to tensorboard """

        with self.train_summary_writer.as_default():
            tf.summary.scalar('loss', self.train_loss.result(), step=step)
            tf.summary.scalar('accuracy', self.train_accuracy.result(), step=step)

        with self.val_summary_writer.as_default():
            tf.summary.scalar('loss', self.val_loss.result(), step=step)
            tf.summary.scalar('accuracy', self.val_accuracy.result(), step=step)

    def train(self):
        logging.info(self.model.summary())
        logging.info('\n================ Starting Training ================')
        self.acc = 0

        for idx, (features, labels) in enumerate(self.ds_train):
            step = idx + 1
            self.train_step(features, labels)
            # logging.info('\nThe {} step is now being implemented.'.format(step))

            if step % self.log_interval == 0:
                # Reset test metrics
                self.val_loss.reset_states()
                self.val_accuracy.reset_states()

                for val_features, val_labels in self.ds_val:
                    self.val_step(val_features, val_labels)

                template = 'Step {}, Loss: {}, Accuracy: {:.2f}%, Test Loss: {}, Validation Accuracy: {:.2f}%'
                logging.info(template.format(step,
                                             self.train_loss.result(),
                                             self.train_accuracy.result() * 100,
                                             self.val_loss.result(),
                                             self.val_accuracy.result() * 100))

                # Write summary to tensorboard
                self.write_scalar_summary(step)

                # Reset train metrics
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()

                yield self.val_accuracy.result().numpy()

            # Save checkpoint
            if step % self.ckpt_interval == 0:
                if self.acc < self.val_accuracy.result():
                    self.acc = self.val_accuracy.result()
                    logging.info(f'Saving checkpoint to {self.run_paths["path_ckpts_train"]}.')
                    path = self.manager.save()
                    print("model saved to %s" % path)

            if step % self.total_steps == 0:
                logging.info(f'Finished training after {step} steps.')

                # Save final checkpoint
                path = self.checkpoint.save(self.run_paths["path_ckpts_train"])
                print("final model saved to %s" % path)

                return self.val_accuracy.result().numpy()

        template = 'Step {}, Loss: {}, Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}'
        logging.info(template.format(step,
                                     self.train_loss.result(),
                                     self.train_accuracy.result() * 100,
                                     self.val_loss.result(),
                                     self.val_accuracy.result() * 100))

        logging.info('\n================ Finished Training ================')


class Example:
    pass
