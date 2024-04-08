from reservoirpy.nodes import Reservoir, LMS
from reservoirpy.observables import nrmse, rsquare
import numpy as np
from reservoirpy.hyper import research
import json
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
import tensorflow as tf
import keras
import keras_tuner

def objective(dataset, config, *, iss, N, sr, lr, alpha, seed):

    x_train, y_train, x_test, y_test= dataset
    # You can access anything you put in the config
    # file from the 'config' parameter.
    instances = config["instances_per_trial"]

    # The seed should be changed across the instances,
    # to be sure there is no bias in the results
    # due to initialization.
    variable_seed = seed

    losses = []
    r2s = []
    for n in range(instances):
        # Build your model given the input parameters
        reservoir = Reservoir(N,
                              sr=sr,
                              lr=lr,
                              input_scaling=iss,
                              seed=variable_seed)

        readout = LMS(alpha=alpha)

        model = reservoir >> readout

        model.train(x_train, y_train)
        predictions = model.run(x_test)

        loss = nrmse(y_test, predictions, norm_value=np.ptp(x_train))
        r2 = rsquare(y_test, predictions)

        # Change the seed between instances
        variable_seed += 1

        losses.append(loss)
        r2s.append(r2)

    # Return a dictionnary of metrics. The 'loss' key is mandatory when
    # using hyperopt.
    return {'loss': np.mean(losses),
            'r2': np.mean(r2s)}

def hyper_tuning(dataset):
    hyperopt_config = {
        "exp": f"config1",  # the experimentation name
        "hp_max_evals": 200,  # the number of differents sets of parameters hyperopt has to try
        "hp_method": "random",  # the method used by hyperopt to chose those sets (see below)
        "seed": 42,  # the random state seed, to ensure reproducibility
        "instances_per_trial": 3,  # how many random ESN will be tried with each sets of parameters
        "hp_space": {  # what are the ranges of parameters explored
            "N": ["choice",300, 500, 900 ],  # the number of neurons is fixed to 300
            "sr": ["loguniform", 1e-2, 2],  # the spectral radius is log-uniformly distributed between 1e-6 and 10
            "lr": ["loguniform", 1e-2, 1],  # idem with the leaking rate, from 1e-3 to 1
            "iss": ["choice", 0.01, 0.1, 1, 5, 10, 50],  # the input scaling is fixed
            "alpha": ["choice", 1e-5],  # and so is the regularization parameter.
            "seed": ["choice", 1234]  # another random seed for the ESN initialization
        }
    }

    # we precautionously save the configuration in a JSON file
    # each file will begin with a number corresponding to the current experimentation run number.
    with open(f"{hyperopt_config['exp']}.config.json", "w+") as f:
        json.dump(hyperopt_config, f)

    best = research(objective, dataset, f"/home/dimimyl/PycharmProjects/AncReservoir/{hyperopt_config['exp']}.config.json", ".")

def build_model(hp):
    model = Sequential()
    model.add(SimpleRNN(900, input_shape=(1, 1), activation='tanh'))
    model.add(Dense(units=1, activation='tanh'))
    model.compile(loss=loss_fn, optimizer=tf.optimizers.Adam(), metrics=['accuracy'], run_eagerly=True)
@keras.saving.register_keras_serializable(package="my_package", name="loss_fn")
def loss_fn(y_true, y_pred):
    sec_path = tf.convert_to_tensor(np.loadtxt("secondaryA2_50.txt", dtype=float))
    sec_path = tf.cast(sec_path, tf.float32)
    global previous_y_pred
    # Initialize buffer if not yet initialized
    if previous_y_pred is None:
        previous_y_pred = tf.Variable(tf.zeros((50,), dtype=tf.float32))
    # Insert current y_pred at the beginning of the buffer
    y_pred_flat = tf.reshape(y_pred, shape=(1,))  # Flatten y_pred to shape (1,)
    previous_y_pred = tf.concat([y_pred_flat, previous_y_pred[:-1]], axis=0)
    # Compute dot product
    y = tf.reduce_sum(tf.multiply(previous_y_pred, sec_path))
    # Compute loss using previous_y_pred
    loss = tf.reduce_mean(tf.square(y_true - y))
    return loss

tuner = keras_tuner.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5)
