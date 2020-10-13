import deepspeech
from deepspeech import Model, version
import numpy as np
import wave

def metadata_to_string(metadata):
    return ''.join(token.text for token in metadata.tokens)

def main():
    ds = Model("model.pbmm")
    ds.enableExternalScorer("scorer.scorer")
    
    fin = wave.open("a.wav", 'rb')
    audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)
    fs_orig = fin.getframerate()
    audio_length = fin.getnframes() * (1/fs_orig)
    fin.close()

    ds.addHotWord("proves",-5000.0)

    #print("STT with Metadata:")
    #print(ds.sttWithMetadata(audio, 1))
    print("\n\nSTT:")
    print(ds.stt(audio))


if __name__ == '__main__':
    main()