import keras
from reservoirpy.nodes import Reservoir, Ridge, LMS
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from tensorflow.keras import models
from sklearn.metrics import r2_score

def esn(x_train, y_train, x_test, y_test):
    reservoir = Reservoir(900, lr=0.36453999372827345, sr=0.7281951867012687, input_scaling=0.9)
    readout=LMS(alpha=1e-5)
    esn_model = reservoir >> readout
    print("Training model...")
    esn_model.train(x_train, y_train)
    (print('Make predictions...'))
    y_pred = esn_model.run(x_train)
    return y_pred
def fxlms(reference_signal,disturbance,sec_path, sec_path_model):
    sec_path = np.loadtxt("secondaryA2_50.txt", dtype=float)
    prim_path = np.loadtxt("primaryA2_50.txt", dtype=float)
    f = 500
    fs = 2000
    time = 200
    time_steps = np.arange(0, time, 1 / fs)
    fl = 900
    sec_path = np.concatenate((sec_path, np.zeros(fl - len(sec_path))))
    sec_path_model = sec_path
    x_buf = np.zeros(fl)
    w = np.zeros(fl)
    y_buf = np.zeros(fl)
    fx_buf = np.zeros(fl)
    error = np.zeros(len(reference_signal))
    step = 0.0005
    for i in range(len(reference_signal)):
        x_buf = np.concatenate(([reference_signal[i]], x_buf[0:fl - 1]))
        y = np.dot(w, x_buf)
        y_buf = np.insert(y_buf, 0, y)
        y_buf = y_buf[0:fl]
        canceling_signal = np.dot(y_buf, sec_path)
        print(canceling_signal)
        error[i] = disturbance[i] + canceling_signal
        fx = np.dot(x_buf, sec_path_model)
        fx_buf = np.insert(fx_buf, 0, fx)
        fx_buf = fx_buf[0:fl]
        ref_norm = np.sum(np.square(x_buf))
        step_norm = step / (0.0000001 + ref_norm)
        w = np.subtract(w, fx_buf * step_norm * error[i])
    return error


previous_y_pred = None
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
    loss1 = tf.reduce_mean(tf.square(y_true - y_pred))
    loss2 = tf.reduce_mean(tf.square(y_true - y))
    loss=loss1+loss2
    return loss
def train_rnn (x_train, y_train):
    model = Sequential()
    model.add(SimpleRNN(900, input_shape=(1, 1), activation='tanh'))
    model.add(Dense(units=1, activation='tanh'))
    model.compile(loss=loss_fn, optimizer=tf.optimizers.Adam(), metrics=['accuracy'], run_eagerly = True)
    print('Training RNN model...')
    model.fit(x_train, y_train, epochs=2, batch_size=1, verbose=2, validation_split=0.2)
    model_json = model.to_json()
    with open('models/rnn_model_mod' + '.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights('models/rnn_model_mod.weights' + '.h5')  # save trained model to file
    print('Saved model to disk!')
    return model
def test_rnn (x_test, y_test):
    json_file = open('models/rnn_model_mod' + '.json', 'r')
    loaded_model_json = json_file.read()  # load model weights
    json_file.close()
    loaded_model = models.model_from_json(loaded_model_json)
    loaded_model.load_weights('models/rnn_model_mod.weights' + '.h5')
    print("Loaded model from disk")
    y_pred = loaded_model.predict(x_test)
    accuracy=r2_score(y_test, y_pred)
    print(accuracy)
    return y_pred
def sec_path_est():
    sec_path=np.array([0, 0, 1, 1.5, -1])
    f=500
    fs=2000
    fl=300
    step=0.001
    modeling_time=200
    time_steps=np.arange(0,modeling_time,1/fs)
    noise=np.random.normal(0,1,len(time_steps))
    b, a = signal.iirfilter(4, Wn=500, fs=fs, btype="low", ftype="butter")
    reference_noise = signal.lfilter(b, a, noise)
    plt.plot(time_steps, reference_noise)
    plt.show()
    output=np.convolve(reference_noise, sec_path, mode='same')
    sec_est=np.zeros(fl)
    error_buffer=np.zeros(len(time_steps))
    ref_buffer=np.zeros(fl)
    for i in range(len(output)):
        ref_buffer = np.concatenate(([reference_noise[i]],ref_buffer[0:fl-1]))
        filter_out = np.dot(sec_est,ref_buffer)
        error_buffer[i]=output[i]-filter_out
        ref_norm=np.sum(np.square(ref_buffer))
        step_norm=step/(0.0001+ref_norm)
        sec_est=np.add(sec_est , ref_buffer*step_norm*error_buffer[i])
    plt.plot(np.arange(fl),sec_est)
    plt.show()
    np.savetxt('/home/dimimyl/PycharmProjects/AncReservoir/secondary_path_estimation.txt', sec_est)
    return sec_est, sec_path
