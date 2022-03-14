import os
import numpy as np
from scipy.fftpack import fft
cnt = 0

dirs = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
fft_all = []
names = []


for direct in dirs:
    waves = [f for f in os.listdir(train_audio_path + direct) if f.endswith('.wav')]
    for wav in waves:
        sample_rate, samples = wavfile.read(train_audio_path + direct + '/' + wav)
        if samples.shape[0] != sample_rate:
            samples = np.append(samples, np.zeros((sample_rate - samples.shape[0], )))
        x, val = custom_fft(samples, sample_rate)
        fft_all = np.append(fft_all, val)
        print(cnt)
        cnt = cnt + 1
        names = np.append(names, direct + '/' + wav)


fft_all = np.array(fft_all)
fft_all

# 데이터 정규화
fft_all = (fft_all - np.mean(fft_all, axis=0)) / np.std(fft_all, axis=0)

# PCA 차원축소
pca = PCA(n_components = 2)
fft_all = pca.fit_transform(fft_all)

# 축소된 데이터 시각화
plt.scatter(x=fft_all[:,0], y=fft_all[:,1], alpha=0.3)
plt.show()

# X축 성분이 800보다 값 찾기
for i in np.where(fft_all[:,0] > 800)[0]:
    print(names[i])