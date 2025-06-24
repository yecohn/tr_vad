from tr_vad.preprocess import binarize_labels


def test_read_rttm(rttm, wav_id):
    df = rttm[rttm["file_id"] == wav_id]


def test_audio(wav):
    wav


def test_binarize_labels(timestamps, sr, wav, label):
    labels = binarize_labels(timestamps, sr, wav)
    first_start_sample = int(timestamps[0]["start"] * sr)
    first_end_sample = int(timestamps[0]["end"] * sr)
    breakpoint()
    assert len(labels) == len(wav)
    assert (
        labels[first_start_sample:first_end_sample].sum()
        == first_end_sample - first_start_sample
    )
    assert labels == label


def test_labels(labels):
    print(labels)


# def test_sample2frame(feature_map, wav):
#     fake_labels = np.zeros(len(wav))
#     frames = sample2frame(fake_labels)
#     assert frames.shape[0] == feature_map.shape[0]


# def test_sample2frame2(feature_map2, wav2):
#     fake_labels = np.zeros(len(wav2))
#     frames = sample2frame(fake_labels)
#     assert frames.shape[0] == feature_map2.shape[0]
