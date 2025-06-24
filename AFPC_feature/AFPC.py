import numpy as np
import torchaudio
import torch
from tr_vad.utils import HParams

from tr_vad.AFPC_feature import base

hparams = HParams()
FS = hparams.sample_rate
NFFT = hparams.n_fft
WINSTEP = hparams.winstep
WINLEN = hparams.winlen
NFILT = hparams.nfilt
NCOEF = hparams.ncoef


def features(
    test_noisy, fs=16000, nfft=512, winlen=0.032, winstep=0.016, nfilt=64, ncoef=22
):

    # Extract MFCC
    mfcc = base.mfcc(
        signal=test_noisy,
        samplerate=fs,
        winlen=winlen,
        winstep=winstep,
        numcep=ncoef,
        nfilt=nfilt,
        nfft=nfft,
        lowfreq=0,
        highfreq=8000,
        preemph=0.97,
        ceplifter=ncoef,
        appendEnergy=False,
    )
    # Extract NSSC
    ssc = base.ssc(
        signal=test_noisy,
        samplerate=fs,
        winlen=winlen,
        winstep=winstep,
        nfilt=nfilt,
        nfft=nfft,
        lowfreq=0,
        highfreq=8000,
        preemph=0.97,
    )
    nssc = base.norm_ssc(
        ssc, nfilt=nfilt, nfft=nfft, samplerate=fs, lowfreq=0, highfreq=8000
    )
    nssc = nssc[:, 0:ncoef]

    # Delta NSSCs
    delta_ssc = base.delta(nssc, 2)
    delta2_ssc = base.delta(delta_ssc, 2)
    nssc_pac = np.concatenate((nssc, delta_ssc, delta2_ssc), axis=1)

    # Delta MFCCs
    delta_mfcc = base.delta(mfcc, 2)
    delta2_mfcc = base.delta(delta_mfcc, 2)
    mfcc_pac = np.concatenate((mfcc, delta_mfcc, delta2_mfcc), axis=1)

    # AFPCS
    AFPC = np.concatenate((mfcc_pac, nssc_pac), axis=1).astype("float32")
    return AFPC


if __name__ == "__main__":

    wav_path = "/home/yehoshua/projects/vad/data/en_example.wav"
    wav, sr = torchaudio.load(wav_path)
    wav = wav[0]
    # start += int(silence * FS)
    # end += int(silence * FS)

    # rescale wav
    wav = wav / torch.abs(wav).max() * 0.999
    out = wav.detach().numpy()
    feature_input = features(
        out, fs=FS, nfft=NFFT, winstep=WINSTEP, winlen=WINLEN, nfilt=NFILT, ncoef=NCOEF
    )[:, :80]
