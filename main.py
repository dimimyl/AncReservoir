import numpy as np
import models
import matplotlib.pyplot as plt
from loadDataset import loadDataset
from plotting import time_plot, power_spectrum, convergence, time_plot_rnn
from hyperparameter_tuning import hyper_tuning
from scipy import signal
from sklearn.metrics import r2_score

setup_parameters = {
                    'data' : 'air',  # 'air', 'hel', 'yac'
                    'tuning' : 1,
                    'rnn': 0,
                    'esn' : 0,
                    'fxlms' : 0,
                    'plot' : 0
                   }

x_data, y_data, x_train, y_train, x_test, y_test, reference_signal, disturbance = loadDataset(setup_parameters)
x_data = np.reshape(x_data,(x_data.shape[0],1))
y_data = np.reshape(y_data,(y_data.shape[0],1))
x_train = np.reshape(x_train,(x_train.shape[0],1))
y_train = np.reshape(y_train,(y_train.shape[0],1))
x_test = np.reshape(x_test,(x_test.shape[0],1))
y_test = np.reshape(y_test,(y_test.shape[0],1))
time = np.arange(len(y_test))
dataset = (x_train, y_train, x_test, y_test)
error_esn = np.zeros(len(y_data))
error_fxlms = np.zeros(len(y_data))

with open('results/ydata_aircraft.txt', 'wb') as f:
    np.savetxt(f, y_data)

# tune esn parameters
if setup_parameters['tuning']:
    hyper_tuning(dataset)

if setup_parameters['rnn']:
    #model=models.rnn_train(x_train,y_train)
    sec_path = np.loadtxt("secondaryA2_50.txt", dtype=float)
    y_pred = models.rnn_test(x_data , y_data)
    sec_path=np.reshape(sec_path,(len(sec_path),1))
    y_pred=np.reshape(y_pred,(len(y_pred),1))
    y=signal.convolve(y_pred, sec_path, mode='same')
    error_rnn=y_data-y_pred
    with open('results/rnn_aircraft_2_no_secpath.txt.txt', 'wb') as f:
        np.savetxt(f, error_rnn)
    with open('results/rnn_aircraft_2_no_secpath.txt', 'rb') as f:
        error_rnn = np.loadtxt(f)
    acc = r2_score(y_data, y_pred)
    print('accuracy:',acc)
    plt.plot(np.arange(len(y_data)), y_data, np.arange(len(y_data)), error_rnn)
    plt.show()

if setup_parameters['esn']:
    model, mse = models.esn_train(x_train, y_train)
    print(' Make prediction using ESN...')
    error_esn, accuracy = models.esn_test(x_data, y_data, model)
    error_esn = error_esn-np.mean(error_esn)
    with open('results/esn_aircraft_2.txt', 'wb') as f:
        np.savetxt(f, error_esn)
    with open('results/mse_esn_aircraft_2.txt', 'wb') as f:
        np.savetxt(f, mse)

if setup_parameters['fxlms']:
    x_data = np.reshape(x_data,(x_data.shape[0],))
    y_data = np.reshape(y_data,(y_data.shape[0],))
    x_train = np.reshape(x_train,(x_train.shape[0],))
    y_train = np.reshape(y_train,(y_train.shape[0],))
    x_test = np.reshape(x_test,(x_test.shape[0],))
    y_test = np.reshape(y_test,(y_test.shape[0],))
    sec_est, sec_path = models.sec_path_est()
    w, mse = models.fxlms(x_train,y_train,sec_path, sec_est)
    y = signal.convolve(x_data, w,  mode='same')
    error_fxlms = y_data-y
    with open('results/fxlms_aircraft_2.txt', 'wb') as f:
        np.savetxt(f, error_fxlms)
    with open('results/mse_fxlms_aircraft_2.txt', 'wb') as f:
        np.savetxt(f, mse)

if setup_parameters['plot']:
    time_plot()
    power_spectrum()
