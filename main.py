import os
import numpy as np
import librosa
from scipy.signal import spectrogram, correlate

# Define paths
sample_path = "sample_to_identify/the_sample.mp3"  # "/mnt/data/the_sample.mp3"
# Update the candidates files list
candidates_files_folder = "candidates_pool"
candidates_files = os.listdir(candidates_files_folder)


# Load sample
sample, sr = librosa.load(sample_path, sr=None)

# Load candidates
candidates = {}
for path in candidates_files:
    path = os.path.join(candidates_files_folder, file)
    song, _ = librosa.load((path), sr=sr)  # Use the same sample rate for all songs
    candidates[path] = song

# Compute spectrogram of the sample
frequencies_sample, times_sample, Sx_sample = spectrogram(
    sample, fs=sr, nperseg=4096, noverlap=2048, detrend=False, scaling="spectrum"
)

# Compute spectrogram of each candidate song
spectrograms_candidates = {}
for name, song in candidates.items():
    frequencies, times, Sx = spectrogram(
        song, fs=sr, nperseg=4096, noverlap=2048, detrend=False, scaling="spectrum"
    )
    spectrograms_candidates[name] = (frequencies, times, Sx)

# Compare the sample's spectrogram with each candidate song's spectrogram
correlations = {}
for name, (frequencies, times, Sx) in spectrograms_candidates.items():
    # Compute the correlation between the sample's spectrogram and the song's spectrogram
    correlation = correlate(Sx_sample, Sx, mode="valid")
    correlations[name] = correlation

# Find the song that has the highest maximum correlation with the sample
max_correlation = max(
    (correlation.max(), name) for name, correlation in correlations.items()
)

print(f"The song that the sample most likely came from is: {max_correlation[1]}")
