import gzip
import streamlit as st
import os
import tempfile
import pickle
import time
from collections import defaultdict, Counter
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from audio_recorder_streamlit import audio_recorder

# --- Configuration Constants (Must match build_index.py exactly) ---
TARGET_SR = 11025
FAN_OUT = 5
MIN_AMPLITUDE_DB = -40
MIN_TIME_DELTA = 0
MAX_TIME_DELTA = 40
FREQ_DELTA = 150

st.set_page_config(page_title="Sonic Signatures", layout="wide", page_icon="🎵")

# ------------------------------------------------------------------------
# Core Fingerprinting Functions
# ------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_precomputed_database():
    """Loads the massive database instantly from the compressed pickle file."""
    db_path = 'database.pkl.gz'
    if not os.path.exists(db_path):
        st.error(f"Critical Error: '{db_path}' not found!")
        st.stop()
    
    with gzip.open(db_path, 'rb') as f: 
        data = pickle.load(f)
    return data['db'], data['sr']

def load_raw_audio_bytes(audio_bytes, target_sr=None):
    """Saves raw bytes from the live microphone to a temp file for librosa."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
        
    try:
        y, sr = librosa.load(tmp_path, sr=target_sr)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return y, sr

def get_peaks(y, sr):
    """Extracts structural local peaks from the query audio."""
    S = librosa.stft(y, n_fft=2048, hop_length=512)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    
    neighborhood_size = (15, 15)
    local_max = maximum_filter(S_db, size=neighborhood_size) == S_db
    
    background = (S_db > MIN_AMPLITUDE_DB)
    peaks_mask = local_max & background
    
    freq_indices, time_indices = np.where(peaks_mask)
    return list(zip(time_indices, freq_indices)), S_db

def generate_hashes(peaks):
    """Pairs query peaks to create matching combinatorial hashes."""
    hashes = []
    num_peaks = len(peaks)
    peaks = sorted(peaks, key=lambda x: x[0])
    
    for i in range(num_peaks):
        anchor_time, anchor_freq = peaks[i]
        pairs_found = 0
        for j in range(i + 1, num_peaks):
            target_time, target_freq = peaks[j]
            dt = target_time - anchor_time
            df = abs(target_freq - anchor_freq)
            
            if MIN_TIME_DELTA <= dt <= MAX_TIME_DELTA and df <= FREQ_DELTA:
                hash_signature = (int(anchor_freq), int(target_freq), int(dt))
                hashes.append((hash_signature, int(anchor_time)))
                
                pairs_found += 1
                if pairs_found >= FAN_OUT: break
            if dt > MAX_TIME_DELTA: break
    return hashes

def predict_song(query_hashes, database):
    """Aligns query hashes against the database to find the true match."""
    matches_per_song = defaultdict(list)
    
    for h, t_query in query_hashes:
        if h in database:
            for song_name, t_db in database[h]:
                offset = t_db - t_query
                matches_per_song[song_name].append(offset)
                
    best_song = None
    max_matches = 0
    best_offsets = []
    
    for song, offsets in matches_per_song.items():
        if offsets:
            most_common_offset, count = Counter(offsets).most_common(1)[0]
            if count > max_matches:
                max_matches = count
                best_song = song
                best_offsets = offsets
                
    return best_song, max_matches, best_offsets

# ------------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------------
st.title("🎵 Sonic Signatures: Live Audio Fingerprinter")
st.markdown("This app matches live audio against a pre-computed database using combinatorial hashing.")

with st.spinner("Mounting database to server memory..."):
    database, master_sr = load_precomputed_database()

st.subheader("🔴 Live Record Mode")
st.markdown("Click the microphone below. Let it record **5 to 10 seconds** of music, then click again to stop.")

# The Microphone Widget
audio_bytes = audio_recorder(
    text="", 
    recording_color="#e83e8c", 
    neutral_color="#6c757d", 
    icon_name="microphone", 
    icon_size="3x"
)

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    
    with st.spinner("Analyzing live recording..."):
        start_time = time.time()
        
        # 1. Process the live audio
        y_query, sr = load_raw_audio_bytes(audio_bytes, target_sr=master_sr)
        
        # 2. Extract features
        query_peaks, S_db = get_peaks(y_query, sr)
        query_hashes = generate_hashes(query_peaks)
        
        # 3. Match against database
        best_song, score, target_offsets = predict_song(query_hashes, database)
        
        elapsed_time = time.time() - start_time
        
        # 4. Display Results
        if best_song and score > 5: # Require at least a few aligned hashes to prevent false positives
            st.success(f"**Identified Song:** {best_song}")
            st.info(f"**Confidence Score:** {score} aligned hashes | **Lookup Time:** {elapsed_time:.2f} seconds")
            
            # --- Visualizations ---
            st.subheader("Acoustic Fingerprint Analysis")
            fig, ax = plt.subplots(1, 3, figsize=(18, 5))
            
            # Plot 1: Spectrogram
            librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis='time', y_axis='hz', ax=ax[0], cmap='magma')
            ax[0].set_title('Query Spectrogram')
            ax[0].set_ylim([0, min(sr/2, 5000)])
            
            # Plot 2: Constellation Map
            times = librosa.frames_to_time([p[0] for p in query_peaks], sr=sr, hop_length=512)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)[[p[1] for p in query_peaks]]
            ax[1].scatter(times, freqs, c='cyan', s=10, alpha=0.7, edgecolors='black')
            ax[1].set_title('Extracted Constellation Peaks')
            ax[1].set_xlabel('Time (s)')
            ax[1].set_ylabel('Frequency (Hz)')
            ax[1].set_ylim([0, min(sr/2, 5000)])
            ax[1].set_facecolor('#2b2b2b')
            
            # Plot 3: Offset Histogram
            ax[2].hist(target_offsets, bins=50, color='royalblue', edgecolor='black')
            ax[2].set_title('Temporal Alignment (Offset Match)')
            ax[2].set_xlabel('Time Offset (Frames)')
            ax[2].set_ylabel('Hash Matches')
            
            st.pyplot(fig)
            
        else:
            st.error("No definitive match found in the database. Ensure the music is loud enough and record a slightly longer clip.")