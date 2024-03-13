import numpy as np
import matplotlib.pyplot as plt

sec_path = np.loadtxt("secondaryA2_50.txt", dtype=float)
sec_path_est = np.loadtxt("secondary_path_estimation.txt", dtype=float)
prim_path = np.loadtxt("primaryA2_50.txt", dtype=float)
f = 500
fs = 2000
time = 200
time_steps = np.arange(0, time, 1 / fs)
reference_signal=np.sin(2*np.pi*f*time_steps)
disturbance = np.convolve(reference_signal, prim_path, mode='same')
fl=250
sec_path=np.concatenate((sec_path,np.zeros(fl-len(sec_path))))
#plt.plot(np.arange(fl),sec_path)
#plt.show()

x_buf=np.zeros(fl)
w=np.zeros(fl)
y_buf=np.zeros(fl)
fx_buf=np.zeros(fl)
error=np.zeros(len(reference_signal))
step=0.5
for i in range(len(reference_signal)):
    x_buf = np.concatenate(([reference_signal[i]], x_buf[0:fl - 1]))
    y=np.dot(w,x_buf)
    y_buf=np.insert(y_buf, 0, y)
    y_buf=y_buf[0:fl]

    canceling_signal=np.dot(y_buf,sec_path)
    print(canceling_signal)
    error[i]=disturbance[i]+canceling_signal
    fx=np.dot(x_buf,sec_path_est)
    fx_buf=np.insert(fx_buf,0,fx)
    fx_buf=fx_buf[0:fl]
    ref_norm = np.sum(np.square(x_buf))
    step_norm = step / (0.0000001 + ref_norm)
    w = np.subtract(w, fx_buf * step_norm * error[i])
plt.plot(np.arange(len(error)),error)
plt.show()
