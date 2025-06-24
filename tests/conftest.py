from pytest import fixture
from torchaudio import load
from utils import RTTMProcessor
from preprocess import extract_timestamps_rttm
from AFPC_feature import base

PATH_WAV = "/home/yehoshua/.cache/huggingface/datasets/MSDWILD/wav/01434.wav"
PATH_WAV2 = "/home/yehoshua/.cache/huggingface/datasets/MSDWILD/wav/01412.wav"
PATH_RTTM = "/home/yehoshua/.cache/huggingface/datasets/MSDWILD/rttms/many.val.rttm"
WAV_ID = 1434
SAMPLING_RATE = 16000


@fixture
def wav():
    sig, sr = load(PATH_WAV)
    return sig.squeeze()


@fixture
def wav2():
    sig, sr = load(PATH_WAV2)
    return sig.squeeze()


@fixture
def wav_id():
    return WAV_ID


@fixture
def rttm():
    df = RTTMProcessor.load_rttm(PATH_RTTM)
    return df


@fixture
def rttm_file():
    return PATH_RTTM


@fixture
def timestamps():
    timestamps = extract_timestamps_rttm(PATH_RTTM, WAV_ID)
    return timestamps


@fixture
def sr():
    return SAMPLING_RATE


@fixture
def feature_map():

    sig, sr = load(PATH_WAV)
    sig = sig.squeeze()
    return base.mfcc(sig)


@fixture
def feature_map2():
    sig, sr = load(PATH_WAV2)
    sig = sig.squeeze()
    return base.mfcc(sig)
