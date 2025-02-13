from pydub import AudioSegment
from pydub.silence import split_on_silence
import torch
import librosa
import numpy as np
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.api import TTS
import soundfile as sf
import os, json
import noisereduce as nr
import argparse

def process_directory_audio(root_dir, tts, device, temperature, fade_time_ms):
    # Traverse the directory and process .txt files
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('text.txt'):
                record_path = os.path.join(subdir, file[:-9])
                manipulate_audio(record_path, tts, device, temperature, fade_time_ms)
                                    
def load_word_from_file(file_path):
    with open(file_path, 'r') as file:
        word = file.read().strip()  # Read and remove any surrounding whitespace/newline
    return word

def offset2sample_conv(offset,sr):
    return int(sr*offset*2/100)
    # 316*2/100 * 16000

def load_audio(file_path):
    """ Load an audio file into a pydub AudioSegment. """
    return AudioSegment.from_file(file_path)

def normalize_audio(audio):
    return audio/np.max(audio)


def trim_silence(audio_segment, silence_thresh=-20, min_silence_len=10):
    """ Trim silence from an AudioSegment. """
    chunks = split_on_silence(audio_segment, 
                              min_silence_len=min_silence_len,
                              silence_thresh=silence_thresh)
    trimmed_audio = chunks[0]
    for chunk in chunks[1:]:
        trimmed_audio += chunk
    return trimmed_audio

def ndarray_to_audiosegment(y, sr):
    """ Convert a librosa ndarray to an AudioSegment. """
    # librosa gives floating point numpy array between -1 and 1, pydub expects int16 array
    y_int = (y * 32767).astype(np.int16)
    audio_segment = AudioSegment(
        y_int.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1
    )
    return audio_segment

def crossfade_audios(audio1, audio2, fade_duration=100):
    """ Crossfade two audio segments. """
    return audio1.append(audio2, crossfade=fade_duration)

def manipulate_audio(record_path, tts, device, temperature, fade_time_ms):
    # if record_path=="/home/ob3942/datasets/VidTIMIT/fadg0/outputs/sa2_with_audio": # TODO delete that
        ##record_path = "/home/ob3942/datasets/VidTIMIT/fadg0/outputs/sa2_with_audio"
    print("Processing the sample:")
    print(record_path)
    audio_path = record_path+'.wav'  # Replace with your file path
    txt_path = record_path+"_text.txt_modified_gpt4o.txt"
    json_path = record_path+"_text_timing.json"
    
    synthesized_audio_path = record_path+"_synthesized.wav"
    output_audio_path = record_path + "_phonemeFake.wav"
    
    audio_sr = 16000
    audio, rate = librosa.load(audio_path, sr=audio_sr)
    
    # Apply noise reduction # TODO we can also give input argument to determine we want noise reduction
    # if reduce_noise:
        #noise_sample = audio[:100]  # First second as noise sample (adjust as needed)
        # audio = nr.reduce_noise(y=audio, y_noise=noise_sample, sr=audio_sr, freq_mask_smooth_hz=2000)#, prop_decrease=0.8)
        # audio = normalize_audio(audio)
        # sf.write("temp_denoised_audio.wav", audio, audio_sr)
        # sf.write("temp_denoised_audio.wav", audio, audio_sr)
      # Check if the text file exists
    if not os.path.exists(txt_path):
        print(f"Text file does not exist: {txt_path}")
        with open("errors.txt", "a") as error_file:
            error_file.write(f"{audio_path}\n")
        return
    word_to_find = load_word_from_file(txt_path).split("|")
    
    old_word = word_to_find[2]
    new_word = word_to_find[1]
    
    # Open timing json file that is obtained with whisper.
    file = open(json_path)
    timing_info_json = json.load(file)
    file.close()
    
    for word_info in timing_info_json:
        # TODO this version does not handle the cases where the wav2vec could not find the word of interest
        # TODO this version does not handle multiple occurances of the changing word but just focuses on the first occurrence
        if word_info['text'].lower().replace(" ", "") == old_word.lower().replace(" ", ""):  # Replace 'your_target_word'
            start_sec_audio = word_info['timestamp'][0]
            end_sec_audio = word_info['timestamp'][1]
            try:
                start_sample_audio = int(start_sec_audio*audio_sr)
                end_sample_audio = int(end_sec_audio*audio_sr)
            except TypeError:
                print(f"Skipping due to TypeError: start_sec_audio={start_sec_audio}, end_sec_audio={end_sec_audio}, audio_sr={audio_sr}")
                return
            # start_sample_audio = offset2sample_conv(word_info['start_offset'],sr=audio_sr)
            # end_sample_audio = offset2sample_conv(word_info['end_offset'],sr=audio_sr)            
            # print(f"Word '{old_word.lower()}' starts at {word_info['start_offset']} and ends at {word_info['end_offset']}")
            start_audio = audio[:start_sample_audio]
            middle_audio = audio[start_sample_audio:end_sample_audio]
            end_audio = audio[end_sample_audio:]
            end_to_cut = len(audio)-end_sample_audio
            print("Synthesizing audio...")
            tts.tts_to_file(text=new_word,
                            file_path=synthesized_audio_path,
                            # speaker_wav="/home/ob3942/temp_denoised_audio.wav", # TODO if denoising the original sample
                            speaker_wav = audio_path,
                            language="en")
            # post-process the audio into the proper format
            output_audio = load_audio(synthesized_audio_path)
            output_audio = output_audio.set_frame_rate(audio_sr)
            output_audio_trimmed = trim_silence(output_audio)
            output_audio_trimmed.export(synthesized_audio_path, format="wav")
            #TODO now i need to utilize the times from beginning end the end
            # output_audio_trimmed_array = output_audio_trimmed.get_array_of_samples()
            # output_audio_trimmed_array_np = np.array(output_audio_trimmed_array)
            #output_audio, rate = librosa.load(wav2lipaudio_path, sr=audio_sr)

            # Crossfade audios for smooth transitions
            start_time_audio_ms = 1000*start_sample_audio/audio_sr
            from_end_time_audio_ms = -1000*end_to_cut/audio_sr
            start_label = -1
            end_label = 0
            final_audio = output_audio_trimmed
            if start_time_audio_ms>fade_time_ms:
                if word_info['text']==timing_info_json[0]["text"]: # do not normalize the initial noise if beginning is empty
                    final_audio = ndarray_to_audiosegment(start_audio, audio_sr)
                else:
                    final_audio = ndarray_to_audiosegment(normalize_audio(start_audio), audio_sr)
                final_audio = crossfade_audios(final_audio, output_audio_trimmed)
                start_label = start_time_audio_ms
            if from_end_time_audio_ms*-1>fade_time_ms:
                if word_info["text"]==timing_info_json[-1]["text"]:
                    audio_segment_end = ndarray_to_audiosegment(end_audio, audio_sr)
                else:
                    audio_segment_end = ndarray_to_audiosegment(normalize_audio(end_audio), audio_sr)
                final_audio = crossfade_audios(final_audio, audio_segment_end)
                end_label=from_end_time_audio_ms
                
            
            # Export the final concatenated and crossfaded audio
            final_audio.export(output_audio_path, format="wav")
            print("DeepFaked audio is generated...")
            wav2lip_audio = final_audio[start_time_audio_ms:from_end_time_audio_ms]
            wav2lip_audio.export(synthesized_audio_path, format="wav") # overwrite the existing file to remove the silences and use wav2lip only for this purpose
            print("Fake audio segment is generated...")
            np.save(record_path+"_label.npy",np.array([start_label, end_label]))
            
    # if word_info['text']==timing_info_json[-1]["text"]:
    #     print("Wav2vec and ground-truth mismatch on the changed word!!!")


def main(args):
    print("Initialization...")
    root_dataset_dir = args.datasetDir
    device=args.device
    modelName = args.synthesisModel
    temperature = args.temperature
    fade_time_ms = args.fadetimems
    tts = TTS(modelName, gpu=True).to(torch.device(device))
    
    process_directory_audio(root_dataset_dir, tts, device, temperature, fade_time_ms)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthesize DeepFake audio snippets.")
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        help="The gpu to be used.",
        default = "cuda:4"
    )
    parser.add_argument(
        "--synthesisModel",
        type=str,
        required=False,
        help="TTS model to transcribe",
        default = "tts_models/multilingual/multi-dataset/xtts_v2"
    )
    parser.add_argument(
        "--datasetDir",
        type=str,
        required=False,
        help="Root directory containing the files to process",
        default ="/home/ugrad-su24/ege/PhonemeFake/gen_ITW"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=False,
        help="hyperparameter of the synthesis function",
        default = 0.8
    )
    parser.add_argument(
        "--fadetimems",
        type=int,
        required=False,
        help="the time in ms for fade in and out of the included audio",
        default = 250
    )
    args = parser.parse_args()
    main(args)