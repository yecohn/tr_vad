import numpy as np
import torch
import torchaudio
from AFPC_feature.AFPC import features
from utils import HParams
from tqdm import tqdm
from glob import glob
from functools import partial
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
import argparse
import os
from utils import RTTMProcessor
import pickle

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


hparams = HParams()
FS = hparams.sample_rate
NFFT = hparams.n_fft
WINSTEP = hparams.winstep
WINLEN = hparams.winlen
NFILT = hparams.nfilt
NCOEF = hparams.ncoef


def find_files(directory, pattern="**/*.WAV"):
    return sorted(glob(os.path.join(directory, pattern), recursive=True))


def preprocess_parallel(
    input_dir: str, feature_dir: str, n_jobs=12, tqdm=lambda x: x, silence=1
) -> list:
    """
    Prepare training/testing dataset.

    Args:
        input_dir (str): Input data directory.
        feature_dir (str): Input data save directory after transformation.
        n_jobs (int, optional): Number of jobs. Defaults to 12.
        tqdm (_type_, optional): tqdm function. Defaults to lambda x: x.
        silence (int, optional): Silence duration. Defaults to 1.

    Returns:
        list: output metadata including (wav_path, feature_filename, time_steps, num_frames, vad_start_time_stamp, vad_end_time_stamp)
    """
    executor = ProcessPoolExecutor(max_workers=n_jobs)
    futures = []
    index = 1

    files = find_files(os.path.join(input_dir))
    print("number of files: ", len(files))
    for wav_path in files:
        text_path = os.path.splitext(wav_path)[0]
        text_parts = text_path.split("/")
        text_path = "/".join(text_parts) + ".PHN"

        with open(text_path, encoding="utf-8") as f:
            lines = f.readlines()
            start = int(lines[1].split(" ")[0])
            end = int(lines[-2].split(" ")[1])
            futures.append(
                executor.submit(
                    partial(
                        _process_AFPC, feature_dir, index, wav_path, start, end, silence
                    )
                )
            )
            index += 1

    return [future.result() for future in tqdm(futures) if future.result() is not None]


def _process_AFPC(feature_dir, index, wav_path, start, end, silence):

    try:
        # Load the audio as numpy array
        wav, _ = torchaudio.load(wav_path)
        wav = wav[0]
    except FileNotFoundError:  # catch missing wav exception
        print(
            "file {} present in csv metadata is not present in wav folder. skipping!".format(
                wav_path
            )
        )
        return None

    start += int(silence * FS)
    end += int(silence * FS)

    # rescale wav
    wav = wav / torch.abs(wav).max() * 0.999
    out = wav.detach().numpy()

    feature_input = features(
        out, fs=FS, nfft=NFFT, winstep=WINSTEP, winlen=WINLEN, nfilt=NFILT, ncoef=NCOEF
    )[:, :80]

    num_frames = feature_input.shape[0]

    out = np.pad(out, (0, NFFT // 2), mode="reflect")
    out = out[: num_frames * int(WINSTEP * FS)]
    time_steps = len(out)

    start = round(start / int(time_steps / num_frames))
    end = round(end / int(time_steps / num_frames))

    feature_filename = "afpc-{}.npy".format(index)
    np.save(
        os.path.join(feature_dir, feature_filename), feature_input, allow_pickle=False
    )

    # Return a tuple describing this training example
    return (wav_path, feature_filename, time_steps, num_frames, start, end)


def write_metadata(hparams, metadata, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for m in metadata:
            f.write("|".join([str(x) for x in m]) + "\n")
    timesteps = sum([int(m[2]) for m in metadata])
    sr = hparams.sample_rate
    hours = timesteps / sr / 3600
    print(
        "Write {} utterances, {} audio timesteps, ({:.2f} hours)".format(
            len(metadata), timesteps, hours
        )
    )
    print(
        "Max audio timesteps length: {:.2f} secs".format(
            (max(m[2] for m in metadata)) / sr,
        )
    )


def preprocess(
    input_folders: str,
    output_folder: str,
    output_name: str = "list.txt",
    n_jobs=cpu_count(),
    silence=1,
):
    """
    Prepare dataset.

    Args:
        input_folders (str): input data folders.
        output_folder (str): output feature folder.
        output_name (str, optional): output summary filename. Defaults to 'list.txt'.
        n_jobs (int, optional): number of jobs. Defaults to cpu_count().
        silence (int, optional): silence duration. Defaults to 1.
    """
    feature_dir = os.path.join(output_folder, "afpc")
    if not os.path.isdir(feature_dir):
        os.makedirs(feature_dir)

    metadata = preprocess_parallel(
        input_folders, feature_dir, n_jobs=n_jobs, tqdm=tqdm, silence=silence
    )
    write_metadata(hparams, metadata, os.path.join(output_folder, output_name))


def binarize_labels(timestamps, sampling_rate, wav: np.array):
    """
    Binarizes time-based labels (timestamps) into a sample-based NumPy array.

    Args:
        timestamps (list of dict): List of dictionaries, where each dict has
                                   'start' and 'end' keys representing time in seconds.
                                   Example: [{'start': 0.5, 'end': 1.2}, {'start': 3.0, 'end': 3.5}]
        sampling_rate (int or float): The sampling rate of the audio (samples per second).
        wav (np.array): The input audio waveform array. Used to determine the total number of samples.

    Returns:
        np.array: A 1D NumPy array of the same length as 'wav', where 1 indicates
                  the presence of a label and 0 indicates no label.
    """
    num_samples = len(wav)
    labels = np.zeros(num_samples, dtype=int)  # Use dtype=int for binary labels

    # Iterate through each timestamp and mark the corresponding sample range
    for timestamp in timestamps:
        start_time = timestamp.get("start")
        end_time = timestamp.get("end")

        # Convert times to sample indices
        # np.floor for start to ensure we don't start marking too early due to float precision
        # np.ceil for end to ensure we include the very last sample of the segment
        # Ensure indices are within the bounds of num_samples
        start_sample = int(np.floor(start_time * sampling_rate))
        end_sample = int(np.ceil(end_time * sampling_rate))

        # Clamp indices to valid array bounds
        start_sample = max(0, start_sample)
        end_sample = min(num_samples, end_sample)

        # Mark the range directly using slicing
        if start_sample < end_sample:  # Ensure the range is valid
            labels[start_sample:end_sample] = 1

    return labels


def sample2frame(samples, sr=16000, win_len=0.032, win_step=0.016):
    """
    we create packets of frames
    window_len: the size of window in ms
    window_step: the step between each window of analysis
    sr: sampling rate
    """
    sample_len = int(win_len * sr)
    sample_step = int(win_step * sr)
    windows = np.lib.stride_tricks.sliding_window_view(samples, window_shape=sample_len)
    return windows[::sample_step]


def extract_wav_labels_from_rttm(rttm_file, wav_dir, save_path=None):
    df = RTTMProcessor.load_rttm(rttm_file)
    sample_ids = df["file_id"].unique()
    timestamps = []
    for sample_id in tqdm(sample_ids):
        dic = dict()
        file_name = (5 - len(str(sample_id))) * "0" + str(sample_id)
        wav_path = os.path.join(wav_dir, f"{file_name}.wav")
        dic["sample_id"] = sample_id
        dic["wav_path"] = wav_path
        wav, sr = torchaudio.load(wav_path)
        wav = wav.squeeze()
        wav = wav / torch.abs(wav).max() * 0.999
        df_sample = df[df["file_id"] == sample_id].sort_values(by="start").reset_index()
        timestamp = RTTMProcessor.extract_timestamps(df_sample)
        labels = binarize_labels(timestamp, sr, wav)
        labels = sample2frame(labels)
        labels = np.where(np.sum(labels, axis=1) / labels.shape[1] > 0.5, 1, 0)
        feature = features(wav)
        offset = feature.shape[0] - labels.shape[0]  # due to padding during stft
        labels = np.pad(labels, pad_width=(0, offset), mode="edge")
        dic["input"] = feature
        dic["labels"] = labels
        timestamps.append(dic)
    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(timestamps, f)
    return timestamps


def main_legacy():
    """
    Use example:
        python preprocess.py '.../TIMIT_augmented/TRAIN' -silence_pad 1"
    """
    parser = argparse.ArgumentParser(description="Data generation for TIMIT dataset.")
    parser.add_argument(
        "input_folder",
        help="Path to the augmented dataset, e.g., <path_to_TIMIT_augmented>",
    )
    parser.add_argument(
        "-silence_pad",
        "--silence_padding",
        type=int,
        default=1,
        help="Silence padding duration in second",
    )

    args = parser.parse_args()

    silence = args.silence_padding
    input_folder = args.input_folder
    output_folder = input_folder.split("/")[:-1]
    output_folder = "/".join(output_folder) + "/train"

    preprocess(
        input_folder,
        output_folder,
        output_name="list.txt",
        n_jobs=cpu_count() - 1,
        silence=silence,
    )
    print("Done")


if __name__ == "__main__":

    train_path_rttm = (
        "/home/yehoshua/.cache/huggingface/datasets/MSDWILD/rttms/few.train.rttm"
    )
    val_path_rttm = (
        "/home/yehoshua/.cache/huggingface/datasets/MSDWILD/rttms/few.val.rttm"
    )

    test_path_rttm = (
        "/home/yehoshua/.cache/huggingface/datasets/MSDWILD/rttms/many.val.rttm"
    )

    wav_dir = "/home/yehoshua/.cache/huggingface/datasets/MSDWILD/wav"
    label_dic = extract_wav_labels_from_rttm(
        rttm_file=val_path_rttm,
        wav_dir=wav_dir,
        save_path="/home/yehoshua/.cache/huggingface/datasets/MSDWILD/preprocessed/test.pkl",
    )
    print(label_dic)
