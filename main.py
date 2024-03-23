import numpy as np
from loadDataset import loadDataset
from models import esn, fxlms, sec_path_est
import matplotlib.pyplot as plt
from hyperparameter_tuning import hyper_tuning

setup_parameters = {'tuning':0,
                  'fxlms':1,
                  'esn' : 1
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
error_esn_buf=np.zeros((len(x_train)))
error_fxlms_buf=np.zeros((len(x_train)))
if setup_parameters['tuning']==1:
    hyper_tuning(dataset)
else:

    #sec_path = np.loadtxt("secondaryA2_50.txt", dtype=float)
    #y_test, y_pred, loss, accuracy = esn(x_train, y_train, x_test, y_test)
    #print(loss, accuracy)
    #antinoise = y_pred
    #plt.plot(time[0:len(y_pred)], y_test, time, y_test-antinoise)
    y_pred_esn = esn(x_data, y_data, x_test, y_test)
    error_esn= y_data-y_pred_esn
    error_esn=error_esn-np.mean(error_esn)





    x_data = np.reshape(x_data,(x_data.shape[0],))
    y_data = np.reshape(y_data,(y_data.shape[0],))
    x_train = np.reshape(x_train,(x_train.shape[0],))
    y_train = np.reshape(y_train,(y_train.shape[0],))
    x_test = np.reshape(x_test,(x_test.shape[0],))
    y_test = np.reshape(y_test,(y_test.shape[0],))
    sec_est, sec_path = sec_path_est()
    error_fxlms=fxlms(x_data,y_data,sec_path, sec_est)
    plt.plot(np.arange(len(y_data)),y_data, np.arange(len(y_data)),error_fxlms, np.arange(len(y_data)),error_esn)
    plt.show()

#sec_path = np.loadtxt("secondaryA2_50.txt", dtype=float)
#y_test, y_pred, loss, accuracy = esn(x_train, y_train, x_test, y_test)
#print(loss, accuracy)


#y_pred=np.reshape(y_pred,(y_pred.shape[0],))[0:30000]
#y_test=np.reshape(y_test,(y_test.shape[0],))[0:30000]


#antinoise = np.convolve(y_pred,sec_path,mode='same')
#print(y_pred.shape,antinoise.shape, y_test.shape)

#plt.plot(time[0:len(y_test)], y_test)
#plt.plot(time[0:len(y_pred)], (y_test-antinoise))
#plt.show()

