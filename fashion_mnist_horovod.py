import time

import tensorflow as tf
import horovod.tensorflow.keras as hvd
from numpy.random import seed

from dataset import load_data_local, build_tf_dataset

hvd.init()

# GPU restriction
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    # tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()+4], 'GPU')

# TF global random seed
tf.random.set_seed(860597652)
# Numpy global random seed
seed(860597652)

if __name__ == '__main__':
    t = int(time.time())

    _batch_size = 128
    _learning_rate = 0.001
    _epochs = 24
    _log_dir = f'./logs/hvd_{t}'
    _ckpt_dir = f'./ckpt/hvd_{t}'
    _model_dir = f'./model/hvd_{t}'

    # Load raw data
    (X_train, y_train), (X_test, y_test) = load_data_local()

    # Horovod split training data
    lpart = len(y_train) // hvd.size()
    i0 = hvd.rank() * lpart
    i1 = (hvd.rank()+1) * lpart
    X_train, y_train = X_train[i0:i1], y_train[i0:i1]
    dset_train = build_tf_dataset(X_train, y_train, _batch_size)
    dset_test = build_tf_dataset(X_test, y_test, _batch_size)

    # Build model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
        tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Horovod adjust learning rate
    _learning_rate = _learning_rate * hvd.size()
    opt = tf.keras.optimizers.Adam(_learning_rate)

    # Horovod distributed optimizer
    opt = hvd.DistributedOptimizer(opt)

    model.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'],
                  experimental_run_tf_function=False)
    
    # Horovod callbacks
    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
        hvd.callbacks.LearningRateWarmupCallback(initial_lr=_learning_rate,
                                                 warmup_epochs=3,
                                                 verbose=1),
    ]

    # Horovod save checkpoints and make log for tensorboard only on worker 0
    if hvd.rank() == 0:
        callbacks.extend([
            tf.keras.callbacks.ModelCheckpoint(_ckpt_dir),
            tf.keras.callbacks.TensorBoard(_log_dir)
        ])

    # Horovod output logs only on worker 0
    verbose = 1 if hvd.rank() == 0 else 0

    # Train
    op = time.time()
    model.fit(dset_train,
              callbacks=callbacks,
              epochs=_epochs,
              verbose=verbose)
    ed = time.time()

    # Horovod test and save model only on worker 0
    if hvd.rank() == 0:
        # Test
        test_loss, test_acc = model.evaluate(dset_test, verbose=2)
        # Save
        model.save(_model_dir)
        # Result
        with open('test.txt', 'a') as f:
            f.write(f'[HVDx{hvd.size()}] loss: {test_loss:.3f} acc: {test_acc:.3f} t: {(ed - op):.3f}\n')
