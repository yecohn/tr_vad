from preprocess import extract_timestamps_rttm, binarize_labels, sample2frame
import numpy as np


def test_read_rttm(rttm, wav_id):
    df = rttm[rttm["file_id"] == wav_id]


def test_audio(wav):
    wav


def test_extract_timestamps_rttm(rttm_file, wav_id):
    timestamps = extract_timestamps_rttm(rttm_file, wav_id)
    assert type(timestamps == list)
    assert len(timestamps) > 0


def test_binarize_labels(timestamps, sr, wav):
    labels = binarize_labels(timestamps, sr, wav)
    first_start_sample = int(timestamps[0]["start"] * sr)
    first_end_sample = int(timestamps[0]["end"] * sr)
    assert len(labels) == len(wav)
    assert (
        labels[first_start_sample:first_end_sample].sum()
        == first_end_sample - first_start_sample
    )
    assert labels[first_end_sample + 1] == 0


# def test_sample2frame(feature_map, wav):
#     fake_labels = np.zeros(len(wav))
#     frames = sample2frame(fake_labels)
#     assert frames.shape[0] == feature_map.shape[0]


# def test_sample2frame2(feature_map2, wav2):
#     fake_labels = np.zeros(len(wav2))
#     frames = sample2frame(fake_labels)
#     assert frames.shape[0] == feature_map2.shape[0]
