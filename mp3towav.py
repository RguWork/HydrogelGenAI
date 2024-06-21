from pydub import AudioSegment
def mp3_to_wav(input_path, output_path):
    sound = AudioSegment.from_mp3(input_path)
    sound.export(output_path, format="wav")

def one_minute(input_path, output_path):
    '''
    saves the second minute of the song.
    chose the second minute because the first minute of the song is usually the build up
    '''
    sound = AudioSegment.from_mp3(input_path)[60000:120000]
    sound.export(output_path, format="mp3")

#mp3_to_wav("/Users/rgu/Desktop/UROPs/UROP4/test/test_og.mp3", "/Users/rgu/Desktop/UROPs/UROP4/test/test_og.wav")

one_minute("/Users/rgu/Desktop/UROPs/UROP4/uncut081.mp3", "/Users/rgu/Desktop/UROPs/UROP4/081.mp3")