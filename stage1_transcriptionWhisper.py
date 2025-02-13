import torch, os, json, argparse
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import librosa
import numpy as np

def load_audio_names(dataset_dir):
    sample_list = []
    for subdir, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.wav'):
                record_path = os.path.join(subdir, file)
                sample_list.append(record_path)
    return sample_list


def main(args):
    print("Initializing...")
    device = args.device
    model_id = args.transcriptionModel
    dataset_dir = args.datasetDir
    
    print("Loading the transcription model...")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    print("Loading the data...")
    samples = load_audio_names(dataset_dir)
    # samples = samples[:4] # TODO used for fast testing of the script
    print("Transcribing...")
    results = pipe(samples, return_timestamps="word", generate_kwargs={"language": "english"}, batch_size=4) # TODO we can extend it to multiple languages
    print("Saving...")
    for sample_idx in range(len(samples)):
        sample = samples[sample_idx]
        result = results[sample_idx]
        txt = result['text']
        time_inf = result['chunks']
        if time_inf[-1]["timestamp"][1] is None:
            audio, sr = librosa.load(sample, sr=16000)
            duration = librosa.get_duration(y=audio, sr=sr)
            timestamp_list = list(time_inf[-1]["timestamp"])
            timestamp_list[1] = np.around(duration, decimals=2)
            time_inf[-1]["timestamp"] = tuple(timestamp_list)
        transcript_file = sample[:-4]+"_text.txt"
        time_inf_file_name = sample[:-4]+"_text_timing.json"
        
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write(txt)
            
        with open(time_inf_file_name, 'w') as file:
            json.dump(time_inf, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe a directory of audio files with Whisper")
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        help="The gpu to be used.",
        default = "cuda:4"
    )
    parser.add_argument(
        "--transcriptionModel",
        type=str,
        required=False,
        help="Huggingface ML model to transcribe",
        default = "openai/whisper-large-v3-turbo"
    )
    parser.add_argument(
        "--datasetDir",
        type=str,
        required=False,
        help="Root directory containing the audio files to process",
        default = "/home/ugrad-su24/ege/PhonemeFake/gen_ITW_minibatch"
    )
    args = parser.parse_args()
    main(args)