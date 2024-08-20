import os
import numpy as np
import pandas as pd
import librosa

def extract_features_for_60_sec(audio_path):
    try:
        y, sr = librosa.load(audio_path, duration=30)
        
        # Double the length by concatenating the audio with itself
        y = np.tile(y, 2)

        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        rms_var = np.var(rms)

        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroid)
        spectral_centroid_var = np.var(spectral_centroid)
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_bandwidth_mean = np.mean(spectral_bandwidth)
        spectral_bandwidth_var = np.var(spectral_bandwidth)
        
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
        zero_crossing_rate_mean = np.mean(zero_crossing_rate)
        zero_crossing_rate_var = np.var(zero_crossing_rate)
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_means = np.mean(mfccs, axis=1)
        mfcc_vars = np.var(mfccs, axis=1)

        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_stft_mean = np.mean(chroma_stft)
        chroma_stft_var = np.var(chroma_stft)

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(rolloff)
        rolloff_var = np.var(rolloff)

        harmony = librosa.effects.harmonic(y)
        harmony_mean = np.mean(harmony)
        harmony_var = np.var(harmony)

        perceptr = librosa.effects.percussive(y)
        perceptr_mean = np.mean(perceptr)
        perceptr_var = np.var(perceptr)

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo)

        features = {
            'rms_mean': rms_mean,
            'rms_var': rms_var,
            'spectral_centroid_mean': spectral_centroid_mean,
            'spectral_centroid_var': spectral_centroid_var,
            'spectral_bandwidth_mean': spectral_bandwidth_mean,
            'spectral_bandwidth_var': spectral_bandwidth_var,
            'zero_crossing_rate_mean': zero_crossing_rate_mean,
            'zero_crossing_rate_var': zero_crossing_rate_var,
            'chroma_stft_mean': chroma_stft_mean,
            'chroma_stft_var': chroma_stft_var,
            'rolloff_mean': rolloff_mean,
            'rolloff_var': rolloff_var,
            'harmony_mean': harmony_mean,
            'harmony_var': harmony_var,
            'perceptr_mean': perceptr_mean,
            'perceptr_var': perceptr_var,
            'tempo': tempo,
        }
        
        for i in range(1, 21):
            features[f'mfcc{i}_mean'] = mfcc_means[i-1]
            features[f'mfcc{i}_var'] = mfcc_vars[i-1]
        
        return features
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def process_audio_files_in_directory(directory_path):
    features_list = []
    
    for root, dirs, files in os.walk(directory_path):
        for file_name in files:
            if file_name.endswith('.wav'):
                file_path = os.path.join(root, file_name)
                features = extract_features_for_60_sec(file_path)
                if features is not None:
                    features['song_name'] = file_name
                    features['file_path'] = file_path
                    features_list.append(features)
                else:
                    print(f"Skipping file: {file_path} due to processing error.")
    
    df = pd.DataFrame(features_list)
    
    #shift song_name to first col
    cols = ['song_name'] + [col for col in df if col != 'song_name']
    df = df[cols]
    
    #extract genre and numeric part for sorting
    df['genre'] = df['song_name'].apply(lambda x: x.split('.')[0])
    df['file_number'] = df['song_name'].apply(lambda x: int(x.split('.')[1]))

    #sort by genre and file_number
    df = df.sort_values(by=['genre', 'file_number']).reset_index(drop=True)
    
    #get rid of the columns used for sorting
    df = df.drop(columns=['genre', 'file_number'])
    
    return df



directory_path = '/Users/rgu/Desktop/UROPs/UROP4/genres_original' 
features_df = process_audio_files_in_directory(directory_path)

features_df.to_csv('extracted_features_60_sec.csv', index=False)
