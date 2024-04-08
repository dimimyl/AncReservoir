import numpy as np
import matplotlib.pyplot as plt
from loadDataset import loadDataset
from models import esn, fxlms, sec_path_est, train_rnn, test_rnn
from plotting import time_plot, power_spectrum
from hyperparameter_tuning import hyper_tuning
from scipy import signal



setup_parameters = {'tuning':0,
                    'only_plot':0,
                    'rnn':1
                   }

x_data, y_data, x_train, y_train, x_test, y_test, reference_signal, disturbance = loadDataset()
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

# tune esn parameters
if setup_parameters['tuning'] == 1:
    hyper_tuning(dataset)

elif setup_parameters['rnn']==1:
    #model=train_rnn(x_train,y_train)
    sec_path = np.loadtxt("secondaryA2_50.txt", dtype=float)
    y_pred = test_rnn(x_data , y_data)
    sec_path=np.reshape(sec_path,(len(sec_path),1))
    y_pred=np.reshape(y_pred,(len(y_pred),1))
    y=signal.convolve(y_pred,sec_path,mode='same')
    error_rnn=y_data-y
    with open('results/rnn_aircraft.txt', 'wb') as f:
        np.savetxt(f, error_rnn)
    with open('results/rnn_aircraft.txt', 'rb') as f:
        error_rnn = np.loadtxt(f)
    time=np.arange(error_rnn.shape[0])
    plt.plot(time, y_data, time, error_rnn)
    plt.show()

else:

    if setup_parameters['only_plot']:
        time_plot(y_data, error_esn, error_fxlms)
        power_spectrum(y_data, error_esn, error_fxlms)

    else:
        y_pred_esn = esn(x_data, y_data, x_test, y_test)
        error_esn= y_data-y_pred_esn
        error_esn=error_esn-np.mean(error_esn)
        with open('results/esn_hel.txt', 'wb') as f:
            np.savetxt(f, error_esn)
        x_data = np.reshape(x_data,(x_data.shape[0],))
        y_data = np.reshape(y_data,(y_data.shape[0],))
        x_train = np.reshape(x_train,(x_train.shape[0],))
        y_train = np.reshape(y_train,(y_train.shape[0],))
        x_test = np.reshape(x_test,(x_test.shape[0],))
        y_test = np.reshape(y_test,(y_test.shape[0],))
        sec_est, sec_path = sec_path_est()
        error_fxlms=fxlms(x_data,y_data,sec_path, sec_est)
        with open('results/fxlms_hel.txt', 'wb') as f:
            np.savetxt(f, error_fxlms)
        with open('results/ydata_hel.txt', 'wb') as f:
            np.savetxt(f, y_data)
        time_plot(y_data,error_esn, error_fxlms)
        power_spectrum(y_data, error_esn, error_fxlms)


