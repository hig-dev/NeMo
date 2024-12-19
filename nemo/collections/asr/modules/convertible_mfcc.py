import torch
import warnings
from typing import Callable, Optional, Sequence, Tuple, Union
from torchaudio.transforms import AmplitudeToDB, MelScale
from torch import Tensor
from torchaudio import functional as F

__all__ = ["ConvertibleMelSpectrogram", "ConvertibleMFCC"]


class ConvertibleMelSpectrogram(torch.nn.Module):
    r"""Create MelSpectrogram for a raw audio signal.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    This is a composition of :py:func:`torchaudio.transforms.Spectrogram`
    and :py:func:`torchaudio.transforms.MelScale`.

    Sources
        * https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe
        * https://timsainb.github.io/spectrograms-mfccs-and-inversion-in-python.html
        * http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

    Args:
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``400``)
        win_length (int or None, optional): Window size. (Default: ``n_fft``)
        hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``win_length // 2``)
        f_min (float, optional): Minimum frequency. (Default: ``0.``)
        f_max (float or None, optional): Maximum frequency. (Default: ``None``)
        pad (int, optional): Two sided padding of signal. (Default: ``0``)
        n_mels (int, optional): Number of mel filterbanks. (Default: ``128``)
        window_fn (Callable[..., Tensor], optional): A function to create a window tensor
            that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
        power (float, optional): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for magnitude, 2 for power, etc. (Default: ``2``)
        normalized (bool, optional): Whether to normalize by magnitude after stft. (Default: ``False``)
        wkwargs (Dict[..., ...] or None, optional): Arguments for window function. (Default: ``None``)
        center (bool, optional): whether to pad :attr:`waveform` on both sides so
            that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
            (Default: ``True``)
        pad_mode (string, optional): controls the padding method used when
            :attr:`center` is ``True``. (Default: ``"reflect"``)
        onesided: Deprecated and unused.
        norm (str or None, optional): If "slaney", divide the triangular mel weights by the width of the mel band
            (area normalization). (Default: ``None``)
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)
        spec_mode (str, optional): 'torchaudio' or 'DFT'. Defaults to 'DFT'.
        dft_mode (str, optional): 'on_the_fly', 'store', 'input'. Defaults to 'store'.
            on_the_fly = Dynamically creates DFT matrix during inference
                model_size = pretty small
                inference_speed = mild overhead to create DFT matrix
                training = mild overhead to create DFT during inference calls
                on-device integration = Easy
            store = Statically creates DFT matrix, uses precomputed matrix
                model_size = largest
                inference_speed = fastest
                training = use this
                on-device integration = Easy

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> transform = transforms.MelSpectrogram(sample_rate)
        >>> mel_specgram = transform(waveform)  # (channel, n_mels, time)

    See also:
        :py:func:`torchaudio.functional.melscale_fbanks` - The function used to
        generate the filter banks.
    """

    __constants__ = ["sample_rate", "n_fft", "win_length", "hop_length", "pad", "n_mels", "f_min"]

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        pad: int = 0,
        n_mels: int = 128,
        window_fn: Callable[..., Tensor] = torch.hann_window,
        power: float = 2.0,
        normalized: bool = False,
        wkwargs: Optional[dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: Optional[bool] = None,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
        spec_mode: str = "DFT",
        dft_mode: str = "store",
    ) -> None:
        super(ConvertibleMelSpectrogram, self).__init__()
        torch._C._log_api_usage_once("torchaudio.transforms.MelSpectrogram")

        if onesided is not None:
            warnings.warn(
                "Argument 'onesided' has been deprecated and has no influence on the behavior of this module."
            )

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.pad = pad
        self.power = power
        self.normalized = normalized
        self.n_mels = n_mels  # number of mel frequency bins
        self.f_max = f_max
        self.f_min = f_min
        from nemo.collections.asr.modules.stft import ConvertibleSpectrogram

        print("ConvertibleSpectrogram config:")
        print("n_fft:", self.n_fft)
        print("win_length:", self.win_length)
        print("hop_length:", self.hop_length)
        print("pad:", self.pad)
        print("window_fn:", window_fn)
        print("power:", self.power)
        print("normalized:", self.normalized)
        print("wkwargs:", wkwargs)
        print("center:", center)
        print("pad_mode:", pad_mode)
        print("onesided:", True)
        # from torchaudio.transforms import Spectrogram
        self.spectrogram = ConvertibleSpectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            pad=self.pad,
            window_fn=window_fn,
            power=self.power,
            normalized=self.normalized,
            wkwargs=wkwargs,
            center=False,
            pad_mode=pad_mode,
            onesided=True,
            spec_mode=spec_mode,
            dft_mode=dft_mode,
        )
        self.mel_scale = MelScale(
            self.n_mels, self.sample_rate, self.f_min, self.f_max, self.n_fft // 2 + 1, norm, mel_scale
        )

    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: Mel frequency spectrogram of size (..., ``n_mels``, time).
        """
        specgram = self.spectrogram(waveform)
        mel_specgram = self.mel_scale(specgram)
        return mel_specgram


class ConvertibleMFCC(torch.nn.Module):
    r"""Create the Mel-frequency cepstrum coefficients from an audio signal.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    By default, this calculates the MFCC on the DB-scaled Mel spectrogram.
    This is not the textbook implementation, but is implemented here to
    give consistency with librosa.

    This output depends on the maximum value in the input spectrogram, and so
    may return different values for an audio clip split into snippets vs. a
    a full clip.

    Args:
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        n_mfcc (int, optional): Number of mfc coefficients to retain. (Default: ``40``)
        dct_type (int, optional): type of DCT (discrete cosine transform) to use. (Default: ``2``)
        norm (str, optional): norm to use. (Default: ``"ortho"``)
        log_mels (bool, optional): whether to use log-mel spectrograms instead of db-scaled. (Default: ``False``)
        spec_mode (str, optional): 'torchaudio' or 'DFT'. Defaults to 'DFT'.
        dft_mode (str, optional): 'on_the_fly', 'store', 'input'. Defaults to 'store'.
            on_the_fly = Dynamically creates DFT matrix during inference
                model_size = pretty small
                inference_speed = mild overhead to create DFT matrix
                training = mild overhead to create DFT during inference calls
                on-device integration = Easy
            store = Statically creates DFT matrix, uses precomputed matrix
                model_size = largest
                inference_speed = fastest
                training = use this
                on-device integration = Easy
        melkwargs (dict or None, optional): arguments for MelSpectrogram. (Default: ``None``)

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> transform = transforms.MFCC(
        >>>     sample_rate=sample_rate,
        >>>     n_mfcc=13,
        >>>     melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        >>> )
        >>> mfcc = transform(waveform)

    See also:
        :py:func:`torchaudio.functional.melscale_fbanks` - The function used to
        generate the filter banks.
    """

    __constants__ = ["sample_rate", "n_mfcc", "dct_type", "top_db", "log_mels"]

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        dct_type: int = 2,
        norm: str = "ortho",
        log_mels: bool = False,
        spec_mode: str = "DFT",
        dft_mode: str = "store",
        melkwargs: Optional[dict] = None,
    ) -> None:
        super(ConvertibleMFCC, self).__init__()
        supported_dct_types = [2]
        if dct_type not in supported_dct_types:
            raise ValueError("DCT type not supported: {}".format(dct_type))
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.dct_type = dct_type
        self.norm = norm
        self.top_db = 80.0
        self.amplitude_to_DB = AmplitudeToDB("power", self.top_db)

        melkwargs = melkwargs or {}
        self.MelSpectrogram = ConvertibleMelSpectrogram(
            sample_rate=self.sample_rate, spec_mode=spec_mode, dft_mode=dft_mode, **melkwargs
        )

        if self.n_mfcc > self.MelSpectrogram.n_mels:
            raise ValueError("Cannot select more MFCC coefficients than # mel bins")
        dct_mat = F.create_dct(self.n_mfcc, self.MelSpectrogram.n_mels, self.norm)
        self.register_buffer("dct_mat", dct_mat)
        self.log_mels = log_mels

    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: specgram_mel_db of size (..., ``n_mfcc``, time).
        """
        mel_specgram = self.MelSpectrogram(waveform)
        if self.log_mels:
            log_offset = 1e-6
            mel_specgram = torch.log(mel_specgram + log_offset)
        else:
            mel_specgram = self.amplitude_to_DB(mel_specgram)

        # (..., time, n_mels) dot (n_mels, n_mfcc) -> (..., n_nfcc, time)
        mfcc = torch.matmul(mel_specgram.transpose(-1, -2), self.dct_mat).transpose(-1, -2)
        return mfcc
