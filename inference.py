import librosa
import os
import matplotlib.pyplot as plt
from tr_vad.utils import bdnn_prediction, get_parameter_number, data_transform
from tr_vad.params import HParams
from tr_vad.VAD_T import VADModel
import numpy as np
from tr_vad.AFPC_feature import AFPC
import torch.nn.functional as F
import torch
import argparse
import contextlib
import time
import csv


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


@contextlib.contextmanager
def changedir():
    old_dir = os.getcwd()
    os.chdir(os.environ.get("PYTHONPATH"))
    yield
    os.chdir(old_dir)


# Function to convert frame-level VAD output to sample-level
def frame2sample(label, w_len, w_step):
    num_frame = len(label)
    total_len = (num_frame - 1) * w_step + w_len
    raw_label = np.zeros(total_len)
    index = 0
    i = 0

    while True:
        if index + w_len >= total_len:
            break
        if i == 0:
            raw_label[index : index + w_len] = label[i]
        else:
            temp_label = label[i]
            raw_label[index : index + w_len] += temp_label

        i += 1
        index += w_step

    raw_label[raw_label >= 1] = 1
    raw_label[raw_label < 1] = 0

    return raw_label


def parse_args():
    parser = argparse.ArgumentParser(description="Speech VAD Inference")
    parser.add_argument(
        "--input_path",
        type=str,
        default="./data_test/[NOISE]SA1_add_sil_SNR(00)_airport.WAV",
        help="Path to the input audio file",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./tr_vad/checkpoint/weights_10_acc_97.09.pth",
        help="Path to the checkpoint file",
    )
    parser.add_argument(
        "--len_blob",
        type=int,
        default=500,
        help="the number of sample for 1 blob to classify, by default 500 for sampling rate of 16k ~31.25 ms",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    hparams = HParams()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DEVICE = "cpu"
    input_path = args.input_path
    len_blob = args.len_blob
    print(f"input_path: {input_path}")

    checkpoint_path = args.checkpoint_path
    print(f"checkpoint_path: {checkpoint_path}")
    model = VADModel(
        dim_in=hparams.dim_in,
        d_model=hparams.d_model,
        units_in=hparams.units_in,
        units=hparams.units,
        layers=hparams.layers,
        P=hparams.P,
        drop_rate=0,
        activation=hparams.activation,
    ).to(DEVICE)

    get_parameter_number(model)
    window_size, unit_size = hparams.w, hparams.u
    with changedir():
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model"])

    waveform, sr = librosa.load(input_path, sr=hparams.sample_rate)
    waveform = waveform / np.abs(waveform).max() * 0.999

    feature_input = AFPC.features(
        waveform,
        fs=sr,
        nfft=hparams.n_fft,
        winstep=hparams.winstep,
        winlen=hparams.winlen,
        nfilt=hparams.nfilt,
        ncoef=hparams.ncoef,
    )[:, :80]
    feature_input = (feature_input - np.mean(feature_input, axis=0)) / (
        np.std(feature_input, axis=0) + 1e-10
    )
    feature_input = torch.as_tensor(feature_input, dtype=torch.float32)
    feature_input = data_transform(
        feature_input,
        window_size,
        unit_size,
        feature_input.min(),
        DEVICE=torch.device("cpu"),
    )
    feature_input = feature_input[window_size:-window_size, :, :]

    start = time.time()
    with torch.inference_mode():
        train_data = feature_input.to(DEVICE)
        postnet_output = model(train_data)
        _, vad = bdnn_prediction(
            F.sigmoid(postnet_output).cpu().detach().numpy(),
            w=window_size,
            u=unit_size,
            threshold=0.4,
        )
    end = time.time()
    lag = end - start

    vad = np.concatenate((np.zeros(hparams.w), vad[:, 0], np.zeros(hparams.w)))
    vad_sample = frame2sample(
        vad,
        int(hparams.sample_rate * hparams.winlen),
        int(hparams.sample_rate * hparams.winstep),
    )
    vad_sample = torch.tensor(vad_sample)
    preds = [int(chunk.sum() > (len_blob / 2)) for chunk in vad_sample.split(len_blob)]
    res = [input_path, lag, *preds]
    header_preds = [str(elem) for elem in np.arange(len(preds))]
    headers = ["input_path", "lag", *header_preds]

    with open(input_path.replace(".wav", "_tr_vad_pred.csv"), "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(headers)
        csv_writer.writerow(res)


if __name__ == "__main__":
    main()
