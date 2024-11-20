# Split dataset
import os
import torchaudio
import soundfile as sf
from tqdm import tqdm
from utility import get_audio_paths

# Split and save function
def split_and_save(file, save_base_path, target_length, sample_rate):
    # Set up paths
    relative_path = os.path.relpath(file, input_path)
    base_filename = os.path.splitext(os.path.basename(file))[0]
    save_dir = os.path.join(save_base_path, os.path.dirname(relative_path))
    os.makedirs(save_dir, exist_ok=True)

    # Load audio files
    audio, sr = torchaudio.load(file)

    # Split into segments of specified duration
    num_segments = audio.size(1) // target_length
    for i in range(num_segments):
        start = i * target_length
        end = start + target_length
        
        segment = audio[:, start:end]

        # Skip segments smaller than target length
        if segment.size(1) < target_length:
            continue

        # Save segments
        segment_path = os.path.join(save_dir, f"{base_filename}_{i + 1}.wav")
        sf.write(segment_path, segment.cpu().numpy().squeeze(), sr)

# Paths for source and target directories
input_path = "/home/woongjib/Projects/Dataset_Crop/SBR_20_Core/FSD50K"  # Change this to your input dataset folder
save_path = "/home/woongjib/Projects/Dataset_Crop/Splits/SBR_20_Core/FSD50K"  # Change this to your desired output folder

# Settings for [splitting]
split_duration = 1  # Change this to 1, 2, or 3 as desired
sample_rate = 48000  
target_length = int(sample_rate * split_duration)

# Get file list
audio_files = sorted(get_audio_paths(input_path, '.wav'))

# Process all files with progress tracking
for file in tqdm(audio_files, total=len(audio_files), desc="Processing files"):
    split_and_save(file, save_path, target_length, sample_rate)

print("Processing completed.")