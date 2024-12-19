# Copyright 2022 Adobe. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

from typing import Callable, Optional, Union

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["ConvertibleSpectrogram"]


def _Create_DFT_matrix_func(n: torch.Tensor, w: torch.Tensor, window: Optional[torch.Tensor] = None):
    W = torch.unsqueeze(w, 1)
    N = torch.unsqueeze(n, 0)
    temp = torch.matmul(W, N)
    Fr = torch.cos(temp)
    Fi = -torch.sin(temp)

    # Apply time-window
    if window is not None:
        DFTr = torch.unsqueeze(torch.mul(window, Fr), 1)
        DFTi = torch.unsqueeze(torch.mul(window, Fi), 1)
    else:
        DFTr = Fr
        DFTi = Fi

    return DFTr, DFTi


class _STFT_Internal(nn.Module):
    """Short-time Fourier Transform internal class"""

    def __init__(
        self,
        dft_mode: str = "on-the-fly",
        n: Optional[torch.Tensor] = None,
        w: Optional[torch.Tensor] = None,
        window: Optional[torch.Tensor] = None,
        padding: int = 0,
    ):
        super(_STFT_Internal, self).__init__()
        self.dft_mode = dft_mode
        self._create_DFT_matrix = _Create_DFT_matrix_func
        self.padding = padding

        if self.dft_mode == "store":
            assert n != None
            assert w != None
            DFTr, DFTi = self._create_DFT_matrix(n, w, window=window)
            self.register_buffer("DFTr", DFTr)
            self.register_buffer("DFTi", DFTi)

    def get_DFT(self):
        if self.dft_mode == "store":
            return self.DFTr, self.DFTi
        else:
            return None, None

    def complex_to_abs(self, real_x, imag_x, power=False):
        """Convert the real and imaginary parts to magnitude or power
        spectrum."""
        S = real_x**2 + imag_x**2
        if not power:
            S = torch.sqrt(S)
        return S

    def forward_input(
        self,
        x: torch.Tensor,
        DFTr: torch.Tensor,
        DFTi: torch.Tensor,
        hop_size: int = 512,
        power: bool = True,
    ):
        # 1D Convolution separately for real and imaginary parts of DFT matrix
        real_x = F.conv1d(x, DFTr, stride=hop_size, padding=self.padding)
        imag_x = F.conv1d(x, DFTi, stride=hop_size, padding=self.padding)
        return self.complex_to_abs(real_x, imag_x, power=power)

    def forward_precomputed(self, x: torch.Tensor, hop_size: int = 512, power: bool = True):

        # 1D Convolution separately for real and imaginary parts of DFT matrix
        real_x = F.conv1d(x, self.DFTr, stride=hop_size, padding=self.padding)
        imag_x = F.conv1d(x, self.DFTi, stride=hop_size, padding=self.padding)
        return self.complex_to_abs(real_x, imag_x, power=power)

    def forward_on_the_fly(
        self,
        x: torch.Tensor,
        n: torch.Tensor,
        w: torch.Tensor,
        hop_size: int = 512,
        power: bool = True,
        window: torch.Tensor = None,
    ):
        DFTr, DFTi = self._create_DFT_matrix(n, w, window=window)
        return self.forward_input(x, DFTr, DFTi, hop_size, power=power)

    def forward(
        self,
        x: torch.Tensor,
        n: torch.Tensor,
        w: torch.Tensor,
        hop_size: int = 512,
        power: bool = True,
        window: Optional[torch.Tensor] = None,
    ):
        """Inference of internal STFT module

        Args:
            x (_type_): input audio signal batch x samples
            n (_type_): DFT n sequence
            w (_type_): DFT omega sequence
            n_fft (int, optional): FFT size. Defaults to 1024.
            hop_size (int, optional): Hop size. Defaults to 512.
            power (bool, optional): _description_. Defaults to True.
            window (_type_, optional): torch tensor of window. Defaults to None.

        Returns:
            _type_: spectrogram
        """
        # Dynamically compute DFT matrix every time (saves model space, wastes CPU)
        return self.forward_precomputed(x, self.DFTr, self.DFTi, hop_size, power=power)


class ConvertibleSpectrogram(nn.Module):
    """Convertible Spectrogram

    This layer computes a specific spectrogram that should be compatible with
    different on-device formats and bit precision. More specifically:

    - Formats:
        - ONNX
        - CoreML

    - Bit Precision:
        - Full (FP32)
        - Mixed
        - Half (FP16)

    To achieve such compatibility, this class has two different `spec_mode`s:

    - "torchaudio": This is compatible with ONNX v17 and above.
    - "DFT": This is compatible with both CoreML and ONNX.

    It should be possible to train using one mode and do inference with
    a different one (i.e., train with TorchAudio and export to CoreML using
    the DFT mode).

    For log-melspectrograms of mixed and half precision, it is recommended
    to use a `top_db` of 65dBs.
    """

    def __init__(
        self,
        n_fft: int = 400,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        pad: int = 0,
        window_fn: Callable[..., Tensor] = torch.hann_window,
        power: Optional[float] = 2.0,
        normalized: Union[bool, str] = False,
        wkwargs: Optional[dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: bool = True,
        spec_mode: str = "DFT",
        dft_mode: str = "store",
    ):
        """_summary_

        Args:
            n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``400``)
            win_length (int or None, optional): Window size. (Default: ``n_fft``)
            hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``win_length // 2``)
            pad (int, optional): Two sided padding of signal. (Default: ``0``)
            window_fn (Callable[..., Tensor], optional): A function to create a window tensor
                that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
            power (float or None, optional): Exponent for the magnitude spectrogram,
                (must be > 0) e.g., 1 for magnitude, 2 for power, etc.
                If None, then the complex spectrum is returned instead. (Default: ``2``)
            normalized (bool or str, optional): Whether to normalize by magnitude after stft. If input is str, choices are
                ``"window"`` and ``"frame_length"``, if specific normalization type is desirable. ``True`` maps to
                ``"window"``. (Default: ``False``)
            wkwargs (dict or None, optional): Arguments for window function. (Default: ``None``)
            center (bool, optional): whether to pad :attr:`waveform` on both sides so
                that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
                (Default: ``True``)
            pad_mode (string, optional): controls the padding method used when
                :attr:`center` is ``True``. (Default: ``"reflect"``)
            onesided (bool, optional): controls whether to return half of results to
                avoid redundancy (Default: ``True``)
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
        """
        super(ConvertibleSpectrogram, self).__init__()

        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.pad = pad
        self.device = "cpu"

        self.power = power
        self.normalized = normalized

        # These are not used but are needed for compatibility
        self.center = center
        self.pad_mode = pad_mode
        self.onesided = onesided

        self.py = np.pi
        self.dft_mode = dft_mode
        self.spec_transf = None
        self.stft = None

        # Store the window scale, if necessary
        self.window_scale = 1

        # Create mode (torchaudio vs. DFT and DTF mode if applicable)
        self.set_mode(spec_mode, dft_mode=dft_mode, window_fn=window_fn, wkwargs=wkwargs)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        if "cpu" in args:
            self.device = "cpu"
        elif "cuda" in args:
            self.device = "cuda"
        return self

    def set_mode(
        self,
        spec_mode: str,
        dft_mode: str = "on_the_fly",
        window_fn: Optional[Callable[..., Tensor]] = None,
        wkwargs: Optional[dict] = None,
    ):
        """Set the DFT mode. See docs above.

        Args:
            spec_mode (str, optional): 'torchaudio' or 'DFT'. Defaults to 'torchaudio'.
            dft_mode (str): 'on_the_fly', 'store', 'input'. Defaults to 'store'.
            coreml (bool, optional): Whether to use a coreml-compatible version, only needed for on_the_fly

        Returns:
            _type_: Real DFT, Imag DFT as torch.Tensor
        """
        self.spec_mode = spec_mode

        # Type of spectrogram to use
        if self.spec_mode == "DFT":
            self.spec_transf = None
            # Create internal variables needed for DFT
            self.dft_mode = dft_mode
            self.stft = None

            # Create and/or store the STFT window
            window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)

            # If the window length doesn't match n_fft, pad or truncate it
            if self.win_length < self.n_fft:
                pad_total = self.n_fft - self.win_length
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                window = F.pad(window, (pad_left, pad_right), mode="constant", value=0.0)
            elif self.win_length > self.n_fft:
                raise ValueError("win_length must be less than or equal to n_fft")

            self.register_buffer("window", window)

            # Clear DFT state
            n = torch.arange(
                0,
                self.n_fft,
                dtype=torch.float32,
                requires_grad=False,
                device=self.device,
            )
            self.register_buffer("n", n)
            w = (2.0 * self.py / self.n_fft) * torch.arange(
                0,
                self.n_fft / 2 + 1,
                dtype=torch.float32,
                requires_grad=False,
                device=self.device,
            )
            self.register_buffer("w", w)

            if self.dft_mode == "on_the_fly":
                self.stft = _STFT_Internal(dft_mode=dft_mode, padding=self.pad)
                # keep w, n, window
                return None, None

            elif self.dft_mode == "store":
                self.stft = _STFT_Internal(
                    dft_mode=dft_mode,
                    n=n,
                    w=w,
                    window=self.window,
                    padding=self.pad,
                )
                # Purge w, n, window (was already turned into DFT matrix)
                self.w = None
                self.n = None
                return self.stft.get_DFT()
            else:
                raise RuntimeError(f"DFT mode f{self.dft_mode} not supported")

        elif self.spec_mode == "torchaudio":
            from torchaudio.transforms import Spectrogram as TorchaudioSpectrogram

            self.stft = None
            self.spec_transf = TorchaudioSpectrogram(
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                pad=self.pad,
                window_fn=window_fn,
                power=self.power,
                normalized=self.normalized,
                wkwargs=wkwargs,
                pad_mode=self.pad_mode,
                center=False,  # must be False for compatibility with DFT
                onesided=True,  # must be True for compatibility with DFT
            )

    def forward(
        self,
        waveform: torch.Tensor,
    ):
        """_summary_

        Args:
            x (torch.Tensor): input audio batch x samples
            DFTr (torch.Tensor, optional): Real-part of DFT matrix. Defaults to None. Only used for DFT input mode.
            DFTi (torch.Tensor, optional): Imag-part of DFT matrix. Defaults to None. Only used for DFT input mode.
            power (bool, optional): power or mag. Defaults to True.
            db (bool, optional): Decibel scale or not. Defaults to False.
            top_db (float, optional): librosa style normalization. Defaults to None.

        Returns:
            _type_: tensor of (batch x mel x frames)
        """
        # Add channel: (batch x channel x samples)
        power = not math.isclose(self.power, 1.0)
        if self.spec_mode == "torchaudio":
            out = self.spec_transf(waveform)
            if not power:
                out = torch.abs(out)
            return out
        elif self.spec_mode == "DFT":
            x = waveform.unsqueeze(1)
            with torch.amp.autocast(device_type=self.device, enabled=False):
                if self.dft_mode == "store":
                    out = self.stft.forward_precomputed(x, hop_size=self.hop_length, power=power)
                elif self.dft_mode == "on_the_fly":
                    out = self.stft.forward_on_the_fly(
                        x,
                        self.n,
                        self.w,
                        hop_size=self.hop_length,
                        power=power,
                        window=self.window,
                    )
                else:
                    raise RuntimeError(f"DFT mode {self.dft_mode} not supported")
            return out
        else:
            raise RuntimeError(f"Unsupported spec_mode {self.spec_mode} " "(supported modes are 'torchaudio', 'DFT')")


if __name__ == "__main__":
    import librosa
    from torchaudio.transforms import Spectrogram as TorchaudioSpectrogram

    spectrogram_ta_orig = TorchaudioSpectrogram(
        n_fft=512,
        win_length=400,
        hop_length=160,
        pad=0,
        window_fn=torch.hann_window,
        power=2.0,
        normalized=False,
        wkwargs=None,
        center=False,  # must be False for compatibility with DFT
        pad_mode="reflect",
        onesided=True,  # must be True for compatibility with DFT
    )

    spectrogram_ta = ConvertibleSpectrogram(
        n_fft=512,
        win_length=400,
        hop_length=160,
        pad=0,
        window_fn=torch.hann_window,
        power=2.0,
        normalized=False,
        wkwargs=None,
        center=False,
        pad_mode="reflect",
        onesided=True,
        spec_mode="torchaudio",
        dft_mode="store",
    )

    spectrogram_dft = ConvertibleSpectrogram(
        n_fft=512,
        win_length=400,
        hop_length=160,
        pad=0,
        window_fn=torch.hann_window,
        power=2.0,
        normalized=False,
        wkwargs=None,
        center=False,
        pad_mode="reflect",
        onesided=True,
        spec_mode="DFT",
        dft_mode="store",
    )

    # waveform shape: (batch x samples)
    waveform_example_file_path = librosa.example("nutcracker")
    waveform_np, sr = librosa.load(waveform_example_file_path, sr=16000)
    waveform_np = waveform_np[16000 : (3 * 16000 + 16000)].astype(np.float32)
    waveforms_np = np.stack([waveform_np, waveform_np], axis=0)
    waveform = torch.tensor(waveforms_np)
    print(f"waveform.shape={waveform.shape}")

    out_ta_orig = spectrogram_ta_orig(waveform)
    print(f"out_ta_orig.shape={out_ta_orig.shape}, out_ta_orig.dtype={out_ta_orig.dtype}")

    out_ta = spectrogram_ta(waveform)
    print(f"out_ta.shape={out_ta.shape}, out_ta.dtype={out_ta.dtype}")

    assert torch.allclose(out_ta_orig, out_ta, atol=1e-5)
    print("torch.allclose(out_ta_orig, out_ta, atol=1e-5) passed")

    out_dft = spectrogram_dft(waveform)
    print(f"out_dft.shape={out_dft.shape}, out_dft.dtype={out_dft.dtype}")

    assert torch.allclose(out_ta_orig, out_dft, atol=1e-5)
    print("torch.allclose(out_ta_orig, out_dft, atol=1e-5) passed")
