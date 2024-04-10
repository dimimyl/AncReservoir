import numpy as np
from sklearn.model_selection import train_test_split

def loadDataset(setup_parameters):
    lim1 = 2000000
    lim2 = lim1 + 60000
    lim3 = 500000
    lim4 = lim3 + 60000
    prim_path=np.array([0, 0, 0, 0, 0, 1, -0.3, 0.2])
    if setup_parameters['data'] == 'air':
        reference_signal = np.loadtxt('Aircraft_Noise_1st_measurement.txt', dtype=float)[lim1:lim2]
    elif setup_parameters['data'] == 'hel':
        reference_signal = np.loadtxt('BO105-Helicopter_flight_MsPoint-A.txt', dtype=float)[lim3:lim4]
    elif setup_parameters['data'] == 'yac':
        reference_signal = np.loadtxt('Yacht_DECK_1900RPM_Noise_02.txt', dtype=float)[lim1:lim2]
    else:
        raise ValueError('Invalid Dataset')
    disturbance = np.array([np.convolve(reference_signal, prim_path, mode='same')])
    disturbance = np.reshape(disturbance, (disturbance.shape[1]))
    y_data = disturbance[1:]
    x_data = reference_signal[0:len(y_data)]
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, shuffle=False, stratify=None)
    return x_data, y_data, x_train, y_train, x_test, y_test, reference_signal, disturbance
