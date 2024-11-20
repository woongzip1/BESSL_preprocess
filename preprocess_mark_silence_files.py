import os
from pydub import AudioSegment
from tqdm import tqdm
from utility import get_audio_paths

def is_silent(audio_segment, silence_threshold=-40.0):
    """Check if an audio segment is silent based on RMS energy."""
    return audio_segment.dBFS < silence_threshold

def search_and_write_silent_files(audio_dirs, log_file, silence_threshold=-40.0, replace_str=None):
    """
    Log silent files from directories and, if specified, generate an additional log with replaced paths.
    - replace_str: A tuple (original, replacement) to transform the GT paths to core paths.
    """
    audiopaths = get_audio_paths(audio_dirs)
    bar = tqdm(audiopaths, desc="Scanning for silent files")
    core_log_file = None
    
    # Open log files for writing
    with open(log_file, 'w') as log:
        if replace_str:
            core_log_file = log_file.replace('.txt', '_core.txt')
            core_log = open(core_log_file, 'w')
        else:
            core_log = None
        
        for path in bar:
            audio = AudioSegment.from_wav(path)
            if is_silent(audio, silence_threshold):
                log.write(f'"{path}"\n')
                print(f"Silent file found in GT dataset: {path}")

                # Write to core log if replacement string provided
                if core_log:
                    core_path = path.replace(replace_str[0], replace_str[1])
                    core_log.write(f'"{core_path}"\n')
                    print(f"Silent file found in Core dataset: {core_path}")

        # Close core log file if opened
        if core_log:
            core_log.close()

def main():
    # Define directories and log files
    audio_dirs = [
        '/home/woongjib/Projects/Dataset_Crop/Splits/GT/VCTK',
        # '/home/woongjib/Projects/Dataset_gt_crop/MUSDB_CORE_M4a'
    ]
    log_file = "silent_files_log.txt"
    
    # Specify replacement strings for GT and core paths
    replace_str = ("/Splits/GT/", "/Splits/SBR_12_Core/")

    # Find and log silent files for both GT and core datasets
    search_and_write_silent_files(audio_dirs, log_file, silence_threshold=-35.0, replace_str=replace_str)

    """
    apply this to delete silence files
    xargs rm < 35_VCTK_log.txt;
    xargs rm < 35_VCTK_12_log.txt;
    xargs rm < 35_VCTK_20_log.txt;

    MUSDB 1s -> -35 dB threshold

    """
    
if __name__ == "__main__":
    main()
