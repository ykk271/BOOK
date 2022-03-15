# 음성 파일 들어보기
import IPython.display as ipd
from scipy.io import wavfile

import warnings

warnings.filterwarnings('ignore')

# 파일 경로 지정
data_dir = 'C:/KAGGLE_DATA/tensorflow-speech-recognition-challenge/'
train_audio_path = data_dir + 'train/train/audio/'

filename = 'yes/00f0204f_nohash_0.wav'

# 음성 데이터 읽기
sample_rate, samples = wavfile.read(str(train_audio_path)+filename)

# player 실행
ipd.Audio(samples, rate=sample_rate) ## 안타깝게 파이참에서 안됨

# 스펙토그램
import numpy as np
from scipy import signal

import matplotlib.pyplot as plt
import seaborn as sns

import librosa
import librosa.display

fig = plt.figure(figsize=(14,8))
ax1 = fig.add_subplot(211)
ax1.set_title('Raw wave of ' + filename)
ax1.set_ylabel('Amplitude')
ax1.plot(np.linspace(0, sample_rate/len(samples), sample_rate), samples)

# 스팩트로그램을 시각화 하는 함수
def specgram(audio, sample_rate, window_size=20,
             step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                            fs=sample_rate,
                                            window='hann',
                                            nperseg=nperseg,
                                            noverlap=noverlap,
                                            detrend=False)
    return freqs, times, spec.T.astype(np.float32)

freqs, times, spectrogram = specgram(samples, sample_rate)
fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(211)
ax1.imshow(spectrogram.T, aspect='auto', origin='lower',
           extent=[times.min(), times.max(), freqs.min(), freqs.max()])
ax1.set_yticks(freqs[::16])
ax1.set_xticks(times[::16])
ax1.set_title('Spectrogram of ' + filename)
ax1.set_ylabel('Freqs in Hz')
ax1.set_xlabel('Seconds')

# log-스펙트로그램
def log_specgram(audio, sample_rate, window_size=20,
             step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                            fs=sample_rate,
                                            window='hann',
                                            nperseg=nperseg,
                                            noverlap=noverlap,
                                            detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32)+eps)

freqs, times, log_spectrogram = log_specgram(samples, sample_rate)
fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(211)
ax1.imshow(log_spectrogram.T, aspect='auto', origin='lower',
           extent=[times.min(), times.max(), freqs.min(), freqs.max()])
ax1.set_yticks(freqs[::16])
ax1.set_xticks(times[::16])
ax1.set_title('Spectrogram of ' + filename)
ax1.set_ylabel('Freqs in Hz')
ax1.set_xlabel('Seconds')


# mel-스펙트로그램을 계산하는 함수
S = librosa.feature.melspectrogram(samples.astype(np.float16), sr=sample_rate, n_mels=128)
log_S = librosa.power_to_db(S, ref=np.max)

# mel-스펙트로그램 시각화
plt.figure(figsize=(12,4))
librosa.display.specshow(log_S, sr=sample_rate, x_axis='time', y_axis='mel')
plt.colorbar(format='%+02.0f dB')
plt.tight_layout()


# MFCC 계산
mfcc = librosa.feature.mfcc(S=log_S, mfcc=13)
delta2_mfcc = librosa.feature.delta(mfcc, order=2)

# MFCC 시각화
plt.figure(figsize=(12,4))
librosa.display.specshow(delta2_mfcc)
plt.ylabel('MFCC coeffs')
plt.xlabel('Time')
plt.colorbar()
plt.tight_layout()

plt.show()

