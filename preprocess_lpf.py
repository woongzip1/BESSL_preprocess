import pdb
import os
import torchaudio as ta
import torch
from tqdm import tqdm
import scipy.signal as signal
import numpy as np

def apply_lowpass_filter(audio, sr, cutoff_freq):
    """Applies a lowpass filter to the audio signal using FFT-based filtering.
    
    Args:
        audio (Tensor): The audio signal to filter.
        sr (int): The sample rate of the audio.
        cutoff_freq (int): The cutoff frequency for the lowpass filter.
    
    Returns:
        Tensor: The filtered audio signal.
    """
    audio_fft = np.fft.fft(audio.squeeze().numpy())
    frequency = np.fft.fftfreq(audio_fft.size, d=1/sr)
    audio_fft[np.abs(frequency) > cutoff_freq] = 0
    filtered_audio = np.fft.ifft(audio_fft).real
    
    return torch.tensor(filtered_audio, dtype=audio.dtype)

def process_audio(input_file, output_file, cutoff_freq, sr):
    """Processes an individual audio file by applying a lowpass filter.
    
    Args:
        input_file (str): Path to the input audio file.
        output_file (str): Path to save the processed audio file.
        cutoff_freq (int): Cutoff frequency for the lowpass filter.
        sr (int): The sample rate to use for the processed audio.
    """
    # Load the audio file
    audio, orig_sr = ta.load(input_file)
    
    # If necessary, resample the audio to the desired sample rate
    if orig_sr != sr:
        resample_transform = ta.transforms.Resample(orig_freq=orig_sr, new_freq=sr)
        audio = resample_transform(audio)
    
    # Apply the lowpass filter
    filtered_audio = apply_lowpass_filter(audio, sr, cutoff_freq)
    
    # Save the processed audio    
    ta.save(output_file, filtered_audio.unsqueeze(0), sr)
    print(f"Processed {input_file} and saved to \n {output_file}")

def process_folder(input_dir, output_dir, cutoff_freq, sr):
    """Processes a folder of audio files, applying a lowpass filter and saving them in the same directory structure.
    
    Args:
        input_dir (str): Path to the directory containing the original audio files.
        output_dir (str): Path to the directory to save the processed audio files.
        cutoff_freq (int): The cutoff frequency for the lowpass filter.
        sr (int): The sample rate to use for the processed audio.
    """
    # Walk through the directory structure
    for root, dirs, files in os.walk(input_dir):
        for file in tqdm(files):
            if file.endswith('.flac') or file.endswith('.wav') or file.endswith('m4a'):
                input_file = os.path.join(root, file)
                
                # Create the output path while preserving directory structure
                relative_path = os.path.relpath(input_file, input_dir)
                output_file = os.path.join(output_dir, relative_path)
                output_folder = os.path.dirname(output_file)
                
                # Create the output folder if it doesn't exist
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                
                # Process the audio file and apply the lowpass filter
                process_audio(input_file, output_file, cutoff_freq, sr)


if __name__ == "__main__":
    import argparse

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Apply lowpass filter to audio files in a directory")
    parser.add_argument("input_dir", type=str, help="Directory containing the original audio files")
    parser.add_argument("output_dir", type=str, help="Directory to save the processed audio files")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate for the processed audio (default: 16000)")
    parser.add_argument("--cutoff", type=int, default=3000, help="Cutoff frequency for the lowpass filter (default: 3000)")

    args = parser.parse_args()

    # Process the input directory
    # python lowpassfilter.py /mnt/hdd/Dataset/MUSDB18_HQ_mono_48kHz_heaac_16kbps/ MUSDB18_CORE/ --sr 48000 --cutoff 4500; 
    # python lowpassfilter.py /mnt/hdd/Dataset/FSD50K_48kHz_heaac_16kbps FSD50K_CORE/ --sr 48000 --cutoff 4500; 
    
    # python lowpassfilter.py /mnt/hdd/Dataset_BESSL/FSD50K_WB_SEGMENT /mnt/hdd/Dataset_BESSL/FSD50K_LPF --sr 48000 --cutoff 4500; 
    # python lowpassfilter.py /mnt/hdd/Dataset_BESSL/MUSDB_WB_SEGMENT /mnt/hdd/Dataset_BESSL/MUSDB_LPF --sr 48000 --cutoff 4500; 

    process_folder(args.input_dir, args.output_dir, args.cutoff, args.sr)

"""
Terminal
python preprocess_lpf.py /home/woongjib/Projects/Dataset_Crop/Splits/GT /home/woongjib/Projects/Dataset_Crop/Splits/GT_LPF_4640 --sr 48000 --cutoff 4640;

"""