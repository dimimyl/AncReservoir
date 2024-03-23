from reservoirpy.nodes import Reservoir, Ridge, LMS
from reservoirpy.observables import nrmse, rsquare
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def esn(x_train, y_train, x_test, y_test):
    reservoir = Reservoir(900, lr=0.36453999372827345, sr=0.7281951867012687)
    #readout = Ridge(ridge=1e-7)
    readout=LMS(alpha=1e-5)
    esn_model = reservoir >> readout
    print("Training model...")
    #esn_model.fit(x_train, y_train, warmup=0)
    error=esn_model.train(x_train, y_train)
    (print('Make predictions...'))
    #x_test=x_test[0:10000]
    #x_test = np.reshape(x_test, (x_test.shape[0],))
    #sec_path = np.loadtxt("secondaryA2_50.txt", dtype=float)
    #x_test=np.convolve(x_test,sec_path,mode='same')
    #x_test=np.reshape(x_test,(len(x_test),1))
    #y_test=y_test[0:10000]
    #print(x_test.shape,y_test.shape)
    #y_pred = esn_model.run(x_test)
    #loss=nrmse(y_test, y_pred)
    #accuracy=rsquare(y_test, y_pred)
    #return(y_test, y_pred, loss, accuracy)
    return error
def fxlms(reference_signal,disturbance,sec_path, sec_path_model):
    #sec_path = np.loadtxt("secondaryA2_50.txt", dtype=float)
    #sec_path_est = np.loadtxt("secondary_path_estimation.txt", dtype=float)
    prim_path = np.loadtxt("primaryA2_50.txt", dtype=float)
    f = 500
    fs = 2000
    time = 200
    time_steps = np.arange(0, time, 1 / fs)
    #reference_signal = np.sin(2 * np.pi * f * time_steps)
    #disturbance = np.convolve(reference_signal, prim_path, mode='same')
    fl = 250
    sec_path = np.concatenate((sec_path, np.zeros(fl - len(sec_path))))
    sec_path_model = sec_path
    # plt.plot(np.arange(fl),sec_path)
    # plt.show()
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
    #plt.plot(np.arange(len(disturbance)), disturbance)
    #plt.plot(np.arange(len(error)), error)

    #plt.show()
    return error
def sec_path_est():
    sec_path = np.loadtxt("secondaryA2_50.txt", dtype=float)
    #sec_path=np.array([0.01, 0.25, 0.5, 1, 0.5, 0.25, 0.01])
    f=500
    fs=2000
    fl=250
    step=0.001
    modeling_time=200
    time_steps=np.arange(0,modeling_time,1/fs)
    #reference_noise=np.sin(2*np.pi*f*time_steps)
    noise=np.random.normal(0,1,len(time_steps))
    #noise = np.random.rand(len(time_steps))
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
