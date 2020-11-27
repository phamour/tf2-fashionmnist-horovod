import argparse
import tensorflow as tf
from numpy.random import seed

from dataset import load_data_local, build_tf_dataset

# GPU restriction
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[4:], 'GPU')

# TF global random seed
tf.random.set_seed(860597652)
# Numpy global random seed
seed(860597652)

tf.keras.backend.manual_variable_initialization(True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TF2-keras Fashion MNIST',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--model-dir', help='Model directory')
    parser.add_argument('--ckpt-dir', help='Checkpoint directory')

    args = parser.parse_args()

    # assert args.model_dir
    assert args.ckpt_dir

    _batch_size = 128

    # Load raw data
    _, (X_test, y_test) = load_data_local()
    dataset = build_tf_dataset(X_test, y_test, _batch_size)

    # Load model
    model = tf.keras.models.load_model(args.ckpt_dir)
    # model = tf.keras.models.load_model(args.model_dir)
    # model.load_weights(args.ckpt_dir)

    # Eval
    model.evaluate(dataset, verbose=2)
