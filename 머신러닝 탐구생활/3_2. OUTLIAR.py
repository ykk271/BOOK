import os
import numpy as np
from scipy.fftpack import fft
cnt = 0

dirs = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
fft_all = []
names = []

# 이상값 찾기
# 푸리에변환 - 주성분 분석
import numpy as np
from scipy.fftpack import fft


def custom_fft(y, fs):
    T = 1.0 / fs
    N = y.shape[0]
    yf = fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    vals = 2.0/N * np.abs(yf[0:N//2])
    return xf, vals



for direct in dirs:
    waves = [f for f in os.listdir(train_audio_path + direct) if f.endswith('.wav')]
    for wav in waves:
        sample_rate, samples = wavfile.read(train_audio_path + direct + '/' + wav)
        if samples.shape[0] != sample_rate:
            samples = np.append(samples, np.zeros((sample_rate - samples.shape[0], )))
        x, val = custom_fft(samples, sample_rate)
        fft_all = np.append(fft_all, val)
        print('{0}: {1}'.format(direct, cnt))
        cnt = cnt + 1
        names = np.append(names, direct + '/' + wav)

print('COMPLETE')

fft_all = np.array(fft_all)
fft_all

# 데이터 정규화
fft_all = (fft_all - np.mean(fft_all, axis=0)) / np.std(fft_all, axis=0)

# PCA 차원축소
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
fft_all = pca.fit_transform(fft_all)

# 축소된 데이터 시각화
plt.scatter(x=fft_all[:,0], y=fft_all[:,1], alpha=0.3)
plt.show()

# X축 성분이 800보다 값 찾기
for i in np.where(fft_all[:,0] > 800)[0]:
    print(names[i])