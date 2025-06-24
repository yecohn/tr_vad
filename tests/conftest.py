from pytest import fixture
from torchaudio import load
from tr_vad.utils import RTTMProcessor
from tr_vad.AFPC_feature import base
import pickle

PATH_WAV = "/home/yehoshua/.cache/huggingface/datasets/MSDWILD/wav/01434.wav"
PATH_WAV2 = "/home/yehoshua/.cache/huggingface/datasets/MSDWILD/wav/01412.wav"
PATH_RTTM = "/home/yehoshua/.cache/huggingface/datasets/MSDWILD/rttms/few.val.rttm"
PATH_LABEL_PKL = (
    "/home/yehoshua/.cache/huggingface/datasets/MSDWILD/preprocessed/val.pkl"
)
WAV_ID = 2759
SAMPLING_RATE = 16000
with open(PATH_LABEL_PKL, "rb") as f:
    val_dic = pickle.load(f)


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
    df = RTTMProcessor.load_rttm(PATH_RTTM)
    df_wav = df[df["file_id"] == WAV_ID].sort_values(by="start").reset_index()
    timestamps = RTTMProcessor.extract_timestamps(df_wav)
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


@fixture
def label():
    return val_dic[0]["labels"]
