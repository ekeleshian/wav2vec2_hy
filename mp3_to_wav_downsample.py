import pydub
import wave
import audioop
import sys
import os
import time


FOLDER_PATH = '/Users/elizabethkeleshian/wav2vec2-huggingface/cv-corpus-7.0-2021-07-21/hy-AM/clips/'
WAV_PATH_32 = "/Users/elizabethkeleshian/wav2vec2-huggingface/cv-corpus-7.0-2021-07-21/hy-AM/wav_clips_32/"
WAV_PATH_16 = "/Users/elizabethkeleshian/wav2vec2-huggingface/cv-corpus-7.0-2021-07-21/hy-AM/wav_clips_16/"


def downsampleWav(src, dst, inrate=48000, outrate=16000, inchannels=1, outchannels=1):
    '''
     src: path to file mp3 need to downsample
     dst: path to file wav after downsample
     inrate: sample rate mp3 file
     outrate: sample rate you want for wav file
     inchannels: num of channels mp3 file 1: mono, 2:stereo
     outchannels: num of channels wav file 1: mono, 2:stereo
    '''
    if not os.path.exists(src):
        print ('Source not found!')
        return False
    try:
        s_read = wave.open(src, 'r')
        s_write = wave.open(dst, 'w')
    except:
        print ('Failed to open files!')
        return False

    n_frames = s_read.getnframes()
    data = s_read.readframes(n_frames)

    try:
        converted = audioop.ratecv(data, 2, inchannels, inrate, outrate, None)
        if outchannels == 1 & inchannels != 1:
            converted[0] = audioop.tomono(converted[0], 2, 1, 0)
    except:
        print ('Failed to downsample wav')
        return False

    try:
        s_write.setparams((outchannels, 2, outrate, 0, 'NONE', 'Uncompressed'))
        s_write.writeframes(converted[0])
    except:
        print ('Failed to write wav')
        return False

    try:
        s_read.close()
        s_write.close()
    except:
        print ('Failed to close wav files')
        return False

    return True


def prepare_downsample():
    print("preparing downsampling of audio files....\n")
    tic = time.perf_counter()
    for file in os.listdir(FOLDER_PATH):
        if file.endswith(".mp3"):
            sound = pydub.AudioSegment.from_mp3(FOLDER_PATH + file)
            sound.export(WAV_PATH_32 + file.replace('.mp3', '.wav'), format='wav')
            downsampleWav(WAV_PATH_32 + file.replace('.mp3', '.wav'), WAV_PATH_16 + file.replace('.mp3', '.wav'), 32000, 16000, 1, 1)  
    toc = time.perf_counter()
    print(f"downsampling completed in {toc-tic:0.4f} seconds. \n")


if __name__ == "__main__":
    prepare_downsample()
