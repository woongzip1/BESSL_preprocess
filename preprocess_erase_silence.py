from pydub import AudioSegment
from utility import get_audio_paths
import os
from tqdm import tqdm
from mutagen import File
import pdb

"""
Input 
GT path: dirpath to erase silence region
RM path: dirpath to save silence-removed files
"""
# Erase silence files
def remove_silence(audio, silence_thresh=-40, min_silence_len=100):
    non_silent_audio = audio.strip_silence(silence_thresh=silence_thresh, 
                                           silence_len=min_silence_len,
                                           )
    return non_silent_audio

def get_total_audio_length(directory):
    total_length = 0
    paths = get_audio_paths(directory, file_extensions=['.wav'])
    bar = tqdm(paths)
    for path in bar:
        if path.endswith(('.mp3','.wav','.flac')):
            audio_file = File(path)
            total_length += audio_file.info.length
    return total_length

def main():
    ############################################################################ Get total audio paths
    gt_path = "/mnt/hdd/Dataset/MUSDB18_HQ_mono_48kHz"
    rm_path = "/home/woongjib/Projects/Dataset_BESSL/SILENCE_RM_GT/MUSDB18" # Save path

    gt_path = "/mnt/hdd/Dataset/FSD50K_48kHz"
    rm_path = "/home/woongjib/Projects/Dataset_BESSL/SILENCE_RM_GT/FSD50K" # Save path

    gt_path = "/mnt/hdd/Dataset/VCTK-Corpus-0.92_crop"
    rm_path = "/home/woongjib/Projects/Dataset_BESSL/SILENCE_RM_GT/VCTK" # Save path
    ############################################################################

    gt_paths = get_audio_paths(gt_path)
    print(len(gt_paths))

    # total_length = get_total_audio_length(gt_path)
    # print(f"Total audio length: {total_length / 3600:.2f} hours")

    # Flac flag
    _,format = os.path.splitext(gt_paths[0])
    flac = True if format=='.flac' else False

    ############################################################################ From given paths, erase silence files and save into new directory
    bar = tqdm(gt_paths)
    for path in bar:
        # Load audio file (wav)
        audio = AudioSegment.from_file(path, format='flac')
        audio_processed = remove_silence(audio, silence_thresh=-40)
        
        if len(audio_processed) == 0:
            continue
        
        if len(audio_processed) < 1100:  # 1000 ms = 1 second
            repeat_count = (1100 // len(audio_processed)) + 1  # Ensure it exceeds 1 second
            audio_processed = audio_processed * repeat_count
            audio_processed = audio_processed[:1100]  # Ensure the length is exactly 1 second
        
        # Save path
        if flac:
            save_path = path.replace(gt_path, rm_path).replace('.flac', '.wav')
        else:
            save_path = path.replace(gt_path, rm_path)
            
        dirs = os.path.dirname(save_path)
        os.makedirs(dirs, exist_ok=True)
        audio_processed.export(save_path, format="wav")

    total_length = get_total_audio_length(rm_path)
    print(f"Processed Total audio length: {total_length / 3600:.2f} hours")
    # print(f"Processed Total audio length: {total_length} hours")

if __name__ == '__main__':
    main()