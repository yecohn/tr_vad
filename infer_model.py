# vad_service.py

import librosa
import os
import numpy as np
import torch
import torch.nn.functional as F
import time
import csv
from torchaudio import load
from .VAD_T import VADModel
from .AFPC_feature import AFPC
from .params import HParams
from .utils import (
    bdnn_prediction,
    data_transform,
)  # Assuming get_parameter_number is not needed for inference instance
import argparse

# This line should ideally be handled by the script that launches your main app,
# or configured more dynamically if you need to choose GPUs.
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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
            # Ensure summation happens safely, avoid float issues
            # If label[i] is already 0 or 1, simple addition is fine.
            # If it's a probability, you might want a different blending.
            raw_label[index : index + w_len] += temp_label

        i += 1
        index += w_step

    # Binarize the summed labels
    raw_label[raw_label >= 1] = 1  # Any overlap means active speech
    raw_label[raw_label < 1] = 0

    return raw_label


class VADInferrer:
    def __init__(
        self,
        checkpoint_path,
        python_path_for_checkpoint=None,
        quantize=False,
        device="cpu",
    ):
        """
        Initializes the VAD model by loading the checkpoint.
        Args:
            checkpoint_path (str): Path to the model checkpoint file.
            python_path_for_checkpoint (str, optional): If the checkpoint loading
                requires a specific PYTHONPATH (e.g., to find VAD_T model definition),
                provide it here. Otherwise, the current script's path is assumed sufficient.
        """
        self.hparams = HParams()
        self.device = torch.device(device)
        self.window_size, self.unit_size = self.hparams.w, self.hparams.u
        self.checkpoint_path = checkpoint_path
        self.device = device

        print(f"Initializing VADInferrer. Loading model from: {self.checkpoint_path}")

        self.model = VADModel(
            dim_in=self.hparams.dim_in,
            d_model=self.hparams.d_model,
            units_in=self.hparams.units_in,
            units=self.hparams.units,
            layers=self.hparams.layers,
            P=self.hparams.P,
            drop_rate=0,
            activation=self.hparams.activation,
        ).to(self.device)

        # Handle the changedir context if PYTHONPATH is needed for checkpoint loading
        # It's better to ensure VAD_T is importable from where your main script runs,
        # or configure PYTHONPATH system-wide if it's a library.
        # This context manager is a workaround if definitions are only found relative
        # to the training script's original location.
        # Consider making VAD_T and AFPC part of your project's Python package/module structure
        # so they are importable directly without changing directories.

        # If python_path_for_checkpoint is provided, use it. Otherwise assume current path works.
        if python_path_for_checkpoint:
            old_pythonpath = os.environ.get("PYTHONPATH")
            os.environ["PYTHONPATH"] = python_path_for_checkpoint
            # Ensure model definitions are importable if they are in this path
            # You might need to add sys.path.insert(0, python_path_for_checkpoint) here too if imports fail

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            # Handle cases where checkpoint might be wrapped (e.g., from DataParallel)
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = (
                    checkpoint  # Assume checkpoint directly contains state_dict
                )

            # Handle DataParallel prefix if model was saved from DataParallel
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = (
                    k[7:] if k.startswith("module.") else k
                )  # remove 'module.' prefix
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)

            if quantize:
                model = self.model
                model_int8 = torch.ao.quantization.quantize_dynamic(
                    model,  # the original model
                    {
                        torch.nn.Linear,
                        torch.nn.Conv1d,
                        torch.nn.Conv2d,
                    },  # a set of layers to dynamically quantize
                    dtype=torch.qint8,
                )  # the target dtype for quantized weights
                self.model = model_int8

            self.model.eval()  # Set model to evaluation mode
            print("VAD model loaded successfully and set to eval mode.")
        except Exception as e:
            print(f"Error loading model checkpoint: {e}")
            raise  # Re-raise to indicate initialization failure
        finally:
            if python_path_for_checkpoint and old_pythonpath is not None:
                os.environ["PYTHONPATH"] = old_pythonpath
            elif python_path_for_checkpoint and old_pythonpath is None:
                del os.environ["PYTHONPATH"]

    def infer_feature(self, feature, len_blob=500):
        feature_input = feature[:, :80]

        feature_input = (feature_input - np.mean(feature_input, axis=0)) / (
            np.std(feature_input, axis=0) + 1e-10
        )
        feature_input = torch.as_tensor(feature_input, dtype=torch.float32)
        feature_input = data_transform(
            feature_input,
            self.window_size,
            self.unit_size,
            feature_input.min(),
            DEVICE=torch.device("cpu"),  # Data transform might run on CPU first
        )
        # feature_input = feature_input[self.window_size : -self.window_size, :, :]

        start_time = time.time()
        with torch.inference_mode():
            train_data = feature_input.to(self.device)
            try:
                postnet_output = self.model(train_data)
            except torch.OutOfMemoryError:
                # print(f"too much on cuda pass sample {audio_path}")
                return 0, 0
            _, vad_frame_level = bdnn_prediction(
                F.sigmoid(postnet_output).cpu().detach().numpy(),
                w=self.window_size,
                u=self.unit_size,
                threshold=0.4,
            )
        end_time = time.time()
        lag = end_time - start_time

        # Post-processing to sample-level
        # The concatenation of zeros (hparams.w) seems to compensate for windowing
        vad_frame_level = np.concatenate(
            (np.zeros(self.hparams.w), vad_frame_level[:, 0], np.zeros(self.hparams.w))
        )

        vad_sample_level = frame2sample(
            vad_frame_level,
            int(self.hparams.sample_rate * self.hparams.winlen),
            int(self.hparams.sample_rate * self.hparams.winstep),
        )
        vad_sample_level = torch.tensor(vad_sample_level)

        # Blob-level prediction
        # Sum of '1's in a blob > half of blob length implies speech
        preds_blob_level = [
            int(chunk.sum() > (len_blob / 2))
            for chunk in vad_sample_level.split(len_blob)
        ]

        return preds_blob_level, lag

    def infer_vad(self, audio_path, len_blob=500):
        """
        Performs VAD inference on a single audio file.
        Args:
            audio_path (str): Path to the input audio file.
            len_blob (int): The number of samples for 1 blob to classify.
        Returns:
            tuple: (preds, lag), where preds is a list of integers (0 or 1)
                   for each blob, and lag is the inference time.
        """
        print(f"Processing audio: {audio_path}")

        # waveform, sr = librosa.load(audio_path, sr=self.hparams.sample_rate)
        waveform, sr = load(audio_path)  # debug loading non normalize
        waveform = waveform / np.abs(waveform).max() * 0.999  # Normalize

        feature_input = AFPC.features(
            waveform,
            fs=sr,
            nfft=self.hparams.n_fft,
            winstep=self.hparams.winstep,
            winlen=self.hparams.winlen,
            nfilt=self.hparams.nfilt,
            ncoef=self.hparams.ncoef,
        )[
            :, :80
        ]  # Assuming :80 is intentional to select first 80 coefficients

        # Normalization (ensure consistency with training)
        feature_input = (feature_input - np.mean(feature_input, axis=0)) / (
            np.std(feature_input, axis=0) + 1e-10
        )
        feature_input = torch.as_tensor(feature_input, dtype=torch.float32)
        feature_input = data_transform(
            feature_input,
            self.window_size,
            self.unit_size,
            feature_input.min(),
            DEVICE=torch.device("cpu"),  # Data transform might run on CPU first
        )
        # feature_input = feature_input[self.window_size : -self.window_size, :, :]

        start_time = time.time()
        with torch.inference_mode():
            train_data = feature_input.to(self.device)
            try:
                postnet_output = self.model(train_data)
            except torch.OutOfMemoryError:
                print(f"too much on cuda pass sample {audio_path}")
                return 0, 0
            _, vad_frame_level = bdnn_prediction(
                F.sigmoid(postnet_output).cpu().detach().numpy(),
                w=self.window_size,
                u=self.unit_size,
                threshold=0.4,
            )
        end_time = time.time()
        lag = end_time - start_time

        # Post-processing to sample-level
        # The concatenation of zeros (hparams.w) seems to compensate for windowing
        vad_frame_level = np.concatenate(
            (np.zeros(self.hparams.w), vad_frame_level[:, 0], np.zeros(self.hparams.w))
        )

        vad_sample_level = frame2sample(
            vad_frame_level,
            int(self.hparams.sample_rate * self.hparams.winlen),
            int(self.hparams.sample_rate * self.hparams.winstep),
        )
        vad_sample_level = torch.tensor(vad_sample_level)

        # Blob-level prediction
        # Sum of '1's in a blob > half of blob length implies speech
        preds_blob_level = [
            int(chunk.sum() > (len_blob / 2))
            for chunk in vad_sample_level.split(len_blob)
        ]

        return preds_blob_level, lag

    def save_results_to_csv(self, audio_path, preds, lag):
        """
        Saves the VAD inference results to a CSV file.
        """
        res = [audio_path, lag, *preds]
        header_preds = [str(elem) for elem in np.arange(len(preds))]
        headers = ["input_path", "lag", *header_preds]

        output_csv_path = audio_path.replace(".wav", "_tr_vad_pred.csv")
        with open(
            output_csv_path, "w", newline=""
        ) as f:  # newline='' important for csv writer
            csv_writer = csv.writer(f)
            csv_writer.writerow(headers)
            csv_writer.writerow(res)
        print(f"Results saved to: {output_csv_path}")


# Optional: Keep the original main function logic for direct execution if needed,
# but now it uses the VADInferrer class.
if __name__ == "__main__":
    # This block allows running vad_service.py directly for testing, similar to original script
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
        default="./checkpoint/weights_10_acc_97.09.pth",
        help="Path to the checkpoint file",
    )
    parser.add_argument(
        "--len_blob",
        type=int,
        default=500,
        help="the number of sample for 1 blob to classify, by default 500 for sampling rate of 16k ~31.25 ms",
    )
    parser.add_argument(
        "--python_path_for_checkpoint",
        type=str,
        default=None,
        help="PYTHONPATH if modules (e.g., VAD_T) are not found.",
    )
    args = parser.parse_args()

    # Create an instance of the inferrer (model loaded once)
    vad_infer = VADInferrer(args.checkpoint_path, args.python_path_for_checkpoint)

    # Perform inference for the specified input
    preds_result, lag_result = vad_infer.infer_vad(args.input_path, args.len_blob)

    # Save results
    vad_infer.save_results_to_csv(args.input_path, preds_result, lag_result)

    print("\n--- Example of processing another file with the same loaded model ---")
    # Replace with another valid audio file path for testing
    another_audio_path = (
        "./data_test/[NOISE]SA1_add_sil_SNR(00)_airport.WAV"  # Use same for example
    )
    if os.path.exists(another_audio_path):
        preds_another, lag_another = vad_infer.infer_vad(
            another_audio_path, args.len_blob
        )
        print(
            f"Processed '{os.path.basename(another_audio_path)}': Lag={lag_another:.4f}s, Preds={preds_another[:5]}..."
        )
    else:
        print(
            f"Warning: '{another_audio_path}' not found, skipping second inference example."
        )
