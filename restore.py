import math
import numpy as np
import matplotlib.pyplot as plt
import pywt
import torch
import torch.nn as nn
import argparse

# create parametr decoder
parser = argparse.ArgumentParser(description='8 methods to denoise a signal')
# add parameters
parser.add_argument('--mean_window_size', type=int, default=20, help='Mean filter window size')
parser.add_argument('--median_window_size', type=int, default=20, help='Median filter window size')
parser.add_argument('--num_NS', type=int, default=100, help='Number of noisy signals to average')
parser.add_argument('--bandpass_low', type=float, default=0.0, help='Low-pass filter cutoff frequency')
parser.add_argument('--bandpass_high', type=float, default=3.0, help='High-pass filter cutoff frequency')
parser.add_argument('--threshold_ratio', type=float, default=0.5, help='Threshold filter amplitude ratio')
parser.add_argument('--wavelet_threshold', type=float, default=0.5, help='Wavelet filter threshold')
parser.add_argument('--std_window_size', type=int, default=20, help='Standard deviation filter window size')
parser.add_argument('--std_threshold', type=float, default=0.5, help='Standard deviation filter threshold')
parser.add_argument('--model_window_size', type=int, default=100, help='Model window size')
# parse parameters
args = parser.parse_args()

n = 1024
h = 2 * np.pi / n
t = np.arange(0, 2 * np.pi, h)


# you can change this signal
S = 1 * np.sin(2 * t) + 1 * np.cos(7 * t) - np.cos(1 * t)  # original signal


RN = 1 * np.random.randn(n, 1)  # random noise
NS = RN.flatten() + S  # noisy signal

# 1.origin S and noisy NS
plt.subplot(331)
plt.title('NS and S')
plt.xlabel('t')
plt.ylabel('Amplitude')
plt.plot(t, NS, 'b-')
plt.plot(t, S, 'r-.')
plt.legend(['NS', 'S'])
plt.grid(True)

# 2.mean filter S
mean_window_size = args.mean_window_size
NS_copy = np.copy(NS)
mean_NS = np.convolve(NS_copy, np.ones(mean_window_size) / mean_window_size, mode='same')
plt.subplot(332)
plt.title('mean_filter | ws = ' + str(mean_window_size))
plt.plot(t, mean_NS, 'b-')
plt.plot(t, S, 'r-.')
plt.legend(['mean', 'S'])
plt.grid(True)


# 3.median filter S
def median_filter(signal, window_size):
    filtered_signal = np.copy(signal)
    half_window = window_size // 2

    for i in range(half_window, len(signal) - half_window):
        window = signal[i - half_window:i + half_window + 1]
        median = np.median(window)
        filtered_signal[i] = median

    return filtered_signal


NS_copy = np.copy(NS)
median_window_size = args.median_window_size
median_NS = median_filter(NS_copy, median_window_size)
plt.subplot(333)
plt.title('median_filter | ws = ' + str(median_window_size))
plt.plot(t, median_NS, 'b-')
plt.plot(t, S, 'r-.')
plt.legend(['median', 'S'])
plt.grid(True)

# 4.Add up to average
num_NS = args.num_NS
NS_sum = np.copy(NS)
for _ in range(num_NS - 1):
    rn = np.random.randn(n, 1)
    ns = S + rn.flatten()
    NS_sum += ns
average_NS = NS_sum / num_NS
plt.subplot(334)
plt.title('average_filter | ns = ' + str(num_NS))
plt.plot(t, average_NS, 'b-')
plt.plot(t, S, 'r-.')
plt.legend(['average', 'S'])
plt.grid(True)

# 5.bandpass filter
low = args.bandpass_low
high = args.bandpass_high
NS_copy = np.copy(NS)
NS_fft = np.fft.fft(NS_copy)
fs = 1 / h  # 采样频率
freq = np.fft.fftfreq(n, d=1 / fs)  # 获取频率分量
# plt.plot(freq, np.abs(NS_fft))
# plt.show()
filter_mask = ((freq >= low) & (freq <= high)) | ((freq >= -high) & (freq <= -low))
NS_fft[~filter_mask] = 0
# plt.plot(freq, np.abs(NS_fft))
# plt.show()
bandpass_NS = np.fft.ifft(NS_fft)
plt.subplot(335)
plt.title('bandpass_filter | l=' + str(low) + ' h=' + str(high))
plt.plot(t, bandpass_NS, 'b-')
plt.plot(t, S, 'r-.')
plt.legend(['bandpass', 'S'])
plt.grid(True)

# 6.threshold filter
ratio = args.threshold_ratio
NS_copy = np.copy(NS)
NS_fft = np.fft.fft(NS_copy)
fs = 1 / h  # 采样频率
freq = np.fft.fftfreq(n, d=1 / fs)  # 获取频率分量
NS_fft_abs = np.abs(NS_fft)
max_amp = np.max(NS_fft_abs)
NS_fft[NS_fft_abs < max_amp * ratio] = 0
threshold_NS = np.fft.ifft(NS_fft)
plt.subplot(336)
plt.title(f'threshold_filter | r={ratio}')
plt.plot(t, threshold_NS, 'b-')
plt.plot(t, S, 'r-.')
plt.legend(['threshold', 'S'])
plt.grid(True)


# 7.wavelet filter
def sgn(num):
    if num > 0.0:
        return 1.0
    elif num == 0.0:
        return 0.0
    else:
        return -1.0


def wavelet_noising(data, threshold):
    # data = new_df
    # data = data.values.T.tolist()  # 将np.ndarray()转为列表
    w = pywt.Wavelet('sym8')  # 选择sym8小波基
    [ca5, cd5, cd4, cd3, cd2, cd1] = pywt.wavedec(data, w, level=5)  # 5层小波分解

    length1 = len(cd1)
    length0 = len(data)

    Cd1 = np.array(cd1)
    abs_cd1 = np.abs(Cd1)
    median_cd1 = np.median(abs_cd1)

    sigma = (1.0 / 0.6745) * median_cd1
    lamda = sigma * math.sqrt(2.0 * math.log(float(length0), math.e))  # 固定阈值计算
    usecoeffs = [ca5]

    # 软硬阈值折中的方法
    a = threshold

    for k in range(length1):
        if abs(cd1[k]) >= lamda:
            cd1[k] = sgn(cd1[k]) * (abs(cd1[k]) - a * lamda)
        else:
            cd1[k] = 0.0

    length2 = len(cd2)
    for k in range(length2):
        if abs(cd2[k]) >= lamda:
            cd2[k] = sgn(cd2[k]) * (abs(cd2[k]) - a * lamda)
        else:
            cd2[k] = 0.0

    length3 = len(cd3)
    for k in range(length3):
        if abs(cd3[k]) >= lamda:
            cd3[k] = sgn(cd3[k]) * (abs(cd3[k]) - a * lamda)
        else:
            cd3[k] = 0.0

    length4 = len(cd4)
    for k in range(length4):
        if abs(cd4[k]) >= lamda:
            cd4[k] = sgn(cd4[k]) * (abs(cd4[k]) - a * lamda)
        else:
            cd4[k] = 0.0

    length5 = len(cd5)
    for k in range(length5):
        if abs(cd5[k]) >= lamda:
            cd5[k] = sgn(cd5[k]) * (abs(cd5[k]) - a * lamda)
        else:
            cd5[k] = 0.0

    usecoeffs.append(cd5)
    usecoeffs.append(cd4)
    usecoeffs.append(cd3)
    usecoeffs.append(cd2)
    usecoeffs.append(cd1)
    recoeffs = pywt.waverec(usecoeffs, w)  # 信号重构
    return recoeffs


threshold = args.wavelet_threshold
NS_copy = np.copy(NS)
wavelet_NS = wavelet_noising(NS_copy, threshold)
plt.subplot(337)
plt.title('wavelet_filter | a = ' + str(threshold))
plt.plot(t, wavelet_NS, 'b-')
plt.plot(t, S, 'r-.')
plt.legend(['wavelet', 'S'])
plt.grid(True)

# 8.std
std_window_size = args.std_window_size
threshold = args.std_threshold
NS_copy = np.copy(NS)

for i in range(std_window_size // 2, len(NS) - std_window_size // 2):
    window = NS[i - std_window_size // 2:i + std_window_size // 2 + 1]
    std_dev = np.std(window)  # 计算窗口内数据的标准差
    if abs(NS[i] - np.mean(window)) > threshold * std_dev:
        NS_copy[i] = np.mean(window)  # 将离群值替换为窗口内数据的均值

plt.subplot(338)
plt.title('std_filter | ws = ' + str(std_window_size) + ' | a = ' + str(threshold))
plt.plot(t, NS_copy, 'b-')
plt.plot(t, S, 'r-.')
plt.legend(['std', 'S'])
plt.grid(True)

# 9.ML
model_window_size = args.model_window_size
model_half_window = model_window_size // 2


class DenoisingModel(nn.Module):
    def __init__(self, input_size, hidden_size1=64, hidden_size2=64):
        super(DenoisingModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.cnv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16 * input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.cnv2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


loaded_model = DenoisingModel(input_size=model_window_size)
loaded_model.load_state_dict(torch.load('model_ws' + str(model_window_size) + '.pth'))
# loaded_model.load_state_dict(torch.load("why.pth"))
X = []
for i in range(model_half_window, len(NS) - model_half_window):
    window = NS[i - model_half_window: i + model_half_window]
    X.append(window)
X = np.array(X)  # every 10 points is a sample

# use model to predict the signal
loaded_model.eval()
with torch.no_grad():
    X_test = torch.FloatTensor(X)
    X_test = X_test.unsqueeze(1)
    denoised_NS = loaded_model(X_test).numpy().flatten()
    denoised_NS = np.concatenate((S[:model_half_window], denoised_NS))
    denoised_NS = np.concatenate((denoised_NS, S[-model_half_window:]))

plt.subplot(339)
plt.title('NN | ws = ' + str(model_window_size))
plt.plot(t, denoised_NS, 'b-')
plt.plot(t, S, 'r-.')
plt.legend(['NN', 'S'])
plt.tight_layout()
plt.show()

# ns = np.copy(NS)
# ns_fft = np.fft.fft(ns)
# fs = 1 / h  # 采样频率
# freq = np.fft.fftfreq(n, d=1 / fs)  # 获取频率分量
# ns_fft_abs = np.abs(ns_fft)
# plt.plot(freq, ns_fft_abs)
# plt.show()
# # filter_mask = ((freq >= low) & (freq <= high)) | ((freq >= -high) & (freq <= -low))
# ns_fft[~filter_mask] = 0
# ns_fft_abs = np.abs(ns_fft)
# plt.plot(freq, ns_fft_abs)
# plt.show()