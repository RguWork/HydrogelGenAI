import os
import ffmpeg

def loudness_normalization(input_filename, output_filename, target_lufs):
    """ 
    given an input file path and output file path and target_lufs, applies loudness normalization on
    the input file to the target lufs and exports it to the output path
    """
    ffmpeg.input(input_filename).audio.filter('loudnorm', I=target_lufs).output(output_filename).run()

def normalize_audios(audio_path, output_path, target_lufs = -14.0):
    """
    given an audio path, output path, and target loudness units full scale (LUFS). target_lufs
    is by default set to -14, which is standard for audio streaming services like spotify or youtube

    applies loudness normalization across all the mp3 files in the given path and exports
    them to the output path as wav files
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for filename in os.listdir(audio_path):
        if filename.lower().endswith('.mp3'):
            output_filename = f"norm_{os.path.splitext(filename)[0]}.wav"  # change the extension to .wav
            loudness_normalization(os.path.join(audio_path, filename), os.path.join(output_path, output_filename), target_lufs)

normalize_audios("/Users/rgu/Desktop/UROPs/UROP4/diego_100_audios", "/Users/rgu/Desktop/UROPs/UROP4/normalized_100_audios")








