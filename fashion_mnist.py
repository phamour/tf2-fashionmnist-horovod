import time

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

if __name__ == '__main__':
    t = int(time.time())

    _batch_size = 128
    _learning_rate = 0.001
    _epochs = 24
    _log_dir = f'./logs/nohvd_{t}'
    _ckpt_dir = f'./ckpt/nohvd_{t}'
    _model_dir = f'./model/nohvd_{t}'

    # Load raw data
    (X_train, y_train), (X_test, y_test) = load_data_local()
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

    opt = tf.keras.optimizers.Adam(_learning_rate)

    model.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(_ckpt_dir),
        tf.keras.callbacks.TensorBoard(_log_dir)
    ]

    # Train
    op = time.time()
    model.fit(dset_train,
              callbacks=callbacks,
              epochs=_epochs,
              verbose=1)
    ed = time.time()

    # Test
    test_loss, test_acc = model.evaluate(dset_test, verbose=2)

    # Save
    model.save(_model_dir)

    # Result
    with open('test.txt', 'a') as f:
        f.write(f'[NoHVD] loss: {test_loss:.3f} acc: {test_acc:.3f} t: {(ed - op):.3f}\n')
