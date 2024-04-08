import numpy as np
from sklearn.model_selection import train_test_split

def loadDataset():
    prim_path=np.array([0, 0, 0, 0, 0, 0, -0.3, 0.2])
    reference_signal = np.loadtxt('Aircraft_Noise_1st_measurement.txt', dtype=float)[0:5000]
    disturbance = np.array([np.convolve(reference_signal, prim_path, mode='same')])
    disturbance = np.reshape(disturbance, (disturbance.shape[1]))
    y_data = disturbance[1:]
    x_data = reference_signal[0:len(y_data)]
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, shuffle=False, stratify=None)
    return x_data, y_data, x_train, y_train, x_test, y_test, reference_signal, disturbance
