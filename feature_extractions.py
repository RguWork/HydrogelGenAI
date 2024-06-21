import numpy as np
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import os
import pandas as pd
import re

def extract_audio_features(file_path):
    """
    given the path of a wav audio file, finds the fundamental frequency, harmonics, bass, and RMS amplitude
    of the audio
    """
    sr, data = read(file_path)
    
    #ensure mono audio
    if len(data.shape) > 1:
        data = data[:, 0]
    
    #apply Fourier Transform and finds the magnitude of the fft
    fft_spectrum = np.fft.rfft(data)
    frequencies = np.fft.rfftfreq(len(data), d=1/sr)
    magnitude = np.abs(fft_spectrum)
    #magnitude = magnitude_to_db(magnitude)
    
    fundamental_freq_index = np.argmax(magnitude)
    fundamental_frequency = frequencies[fundamental_freq_index]
    
    harmonics_indices = [i for i in range(len(frequencies)) if frequencies[i] % fundamental_frequency < fundamental_frequency * 0.05]
    #if the frequency is some multiple of the fundamental frequency with some small error it should be a harmonic frequency
    harmonics_frequencies = frequencies[harmonics_indices]
    harmonics_magnitude = magnitude[harmonics_indices]

    bass_indices = np.where((frequencies >= 60) & (frequencies <= 250))[0]
    bass_frequencies = frequencies[bass_indices]
    bass_magnitude = magnitude[bass_indices]


    rms_amplitude = calculate_rms(data)
    window_size = int(0.01 * sr) #10 ms window
    windowed_rms_amplitude = calculate_windowed_rms(data, window_size)
    rms_time = np.linspace(0, len(windowed_rms_amplitude) / sr, num=len(windowed_rms_amplitude)) #time vector for RMS plot

    
    # plt.figure(figsize=(14, 7))
    # plt.subplot(311)
    # plt.title('Frequency Spectrum')
    # plt.plot(frequencies, magnitude)
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Magnitude')
    # plt.xlim(0, sr // 2)

    # plt.subplot(312)
    # plt.title('Harmonics')
    # plt.stem(harmonics_frequencies, harmonics_magnitude, 'r')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Magnitude')
    # plt.xlim(0, sr // 2)

    # plt.subplot(313)
    # plt.title('Bass Frequencies')
    # plt.plot(bass_frequencies, bass_magnitude, 'g')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Magnitude')
    # plt.xlim(60, 250)

    # plt.figure(figsize=(10, 4))
    # plt.plot(rms_time, windowed_rms_amplitude)
    # plt.title('RMS Amplitude')
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('RMS Amplitude')
    
    # plt.tight_layout()
    # plt.show()
    
    #print(harmonics_magnitude, bass_frequencies, max(harmonics_magnitude), max(bass_magnitude))

    print('Testing Harmonic and Bass Magnitudes')
    print('Median Magnitudes:', np.median(harmonics_magnitude), np.median(bass_magnitude))
    print('\n')
    return fundamental_frequency, np.mean(harmonics_magnitude), np.mean(bass_magnitude), rms_amplitude

def calculate_windowed_rms(data, window_size):
    """
    calculate RMS amplitude with a given window size
    this is mainly for graph visualization, also probably useful
    if the song amplitude fluctuates a lot
    """
    window = np.ones(window_size) / window_size
    rms = np.sqrt(np.convolve(data**2, window, mode='valid'))
    return rms

def calculate_rms(data):
    return np.sqrt(np.mean(data**2))

def magnitude_to_db(magnitude, ref=1.0):
    return 20 * np.log10(np.maximum(magnitude, 1e-20) / ref)


file_path = "/Users/rgu/Desktop/UROPs/UROP4/normalized_100_audios"



audio_file = []
fundamental_freqs = []
rms_amplis= []

for filename in os.listdir(file_path):
    if filename.lower().endswith('.wav'):
        print("File name:", filename)
        f0, harmonics, bass, rms_amplitude = extract_audio_features(os.path.join(file_path,filename))
        print("Fundamental Frequency:", f0)
        print("Average Harmonics Magnitude:", harmonics) #this and bass are causing problems, way too high
        print("Average Bass Magnitude:", bass)
        print("Average RMS Amplitude:", rms_amplitude, '\n')

        audio_number = int(re.search(r'\d+', filename).group())
        audio_file.append(audio_number)
        fundamental_freqs.append(f0)
        rms_amplis.append(rms_amplitude)


variables = pd.DataFrame({
    'song_id': audio_file,
    'fundamental_frequency': fundamental_freqs,
    'RMS_amplitude': rms_amplis
})

variables = variables.sort_values(by='song_id')
variables.to_csv("/Users/rgu/Desktop/UROPs/UROP4/audio_variables.csv", index=False)


#below is debugging code
# _, a = read('/Users/rgu/Desktop/UROPs/UROP4/normalized_100_audios/norm_001.wav')
# _, b = read('/Users/rgu/Desktop/UROPs/UROP4/normalized_100_audios/norm_082.wav')
# _, c = read('/Users/rgu/Desktop/UROPs/UROP4/normalized_100_audios/norm_083.wav')

# print(calculate_rms(a))
# print(calculate_rms(b))
# print(calculate_rms(c))

#print(np.mean(a**2))
# def calculate_rms(data):
#     return np.sqrt(np.mean(data**2))
# for ind,i in enumerate(a**2):
#     if i[0] <= 0 or i[1] <= 0:
#         print(ind, i)
# print(a[164679], a[164679]**2) #wtf [-7851 -9357] [-31175  -2647]
# print(a.dtype, '\n') 

# a_float = a.astype(np.float64)
# squared = a_float**2 

# print(a_float[164679], '\n')
# print(squared[164679])

# print('rms_fixed:', calculate_rms(a_float))


# extract_audio_features('/Users/rgu/Desktop/UROPs/UROP4/normalized_100_audios/norm_081.wav')
# extract_audio_features('/Users/rgu/Desktop/UROPs/UROP4/normalized_100_audios/norm_000.wav')