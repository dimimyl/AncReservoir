import numpy as np
from loadDataset import loadDataset
from models import esn, fxlms, sec_path_est
import matplotlib.pyplot as plt

x_train, y_train, x_test, y_test, reference_signal, disturbance = loadDataset()
x_train = np.reshape(x_train,(x_train.shape[0],1))
y_train = np.reshape(y_train,(y_train.shape[0],1))
x_test = np.reshape(x_test,(x_test.shape[0],1))
y_test = np.reshape(y_test,(y_test.shape[0],1))
time = np.arange(len(y_test))
dataset = (x_train, y_train, x_test, y_test)
#x_train = np.reshape(x_train,(x_train.shape[0],))
#y_train = np.reshape(y_train,(y_train.shape[0],))
#x_test = np.reshape(x_test,(x_test.shape[0],))
#y_test = np.reshape(y_test,(y_test.shape[0],))
#sec_est, sec_path = sec_path_est()
#fxlms(x_train,y_train,sec_path, sec_est)
#hyper_tuning(dataset)
sec_path = np.loadtxt("secondaryA2_50.txt", dtype=float)
y_test, y_pred, loss, accuracy = esn(x_train, y_train, x_test, y_test)
print(loss, accuracy)


y_pred=np.reshape(y_pred,(y_pred.shape[0],))[0:30000]
y_test=np.reshape(y_test,(y_test.shape[0],))[0:30000]
antinoise=y_pred

#antinoise = np.convolve(y_pred,sec_path,mode='same')
#print(y_pred.shape,antinoise.shape, y_test.shape)
#plt.plot(time[0:len(y_pred)], y_test[0:500], time, y_test-antinoise)
plt.plot(time[0:len(y_test)], y_test)
plt.plot(time[0:len(y_pred)], (y_test-antinoise))
plt.show()

