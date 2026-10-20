"""Streaming complex-baseband FM for software-defined communication experiments.

This module has no file, plot, device, or command-line dependencies.  The
existing real-passband functions in modulation.py and demodulation.py remain
available for the original experiments.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, firwin, lfilter, sosfilt

from .channel import awgn


def _vector(x, *, complex_input=False):
    a = np.asarray(x)
    if a.ndim != 1:
        raise ValueError("expected a one-dimensional block")
    if complex_input:
        if not np.iscomplexobj(a):
            raise ValueError("IQ input must be complex")
        a = a.astype(np.complex128, copy=False)
    else:
        if np.iscomplexobj(a):
            raise ValueError("message must be real")
        a = a.astype(np.float64, copy=False)
    if not np.all(np.isfinite(a)):
        raise ValueError("block contains NaN or Inf")
    return a


class ComplexFMModulator:
    """z[n] = exp(j*(phase[n-1] + 2*pi*deviation*m[n]/fs)).

    The caller supplies a fixed normalized message scale; block-wise peak
    normalization would change the physical deviation at every boundary.
    """

    def __init__(self, sample_rate, freq_deviation, initial_phase=0.0):
        self.sample_rate = float(sample_rate)
        self.freq_deviation = float(freq_deviation)
        self.initial_phase = float(initial_phase)
        if not all(map(np.isfinite, (self.sample_rate, self.freq_deviation,
                                      self.initial_phase))):
            raise ValueError("parameters must be finite")
        if not (self.sample_rate > 0 and 0 < self.freq_deviation < 0.45*self.sample_rate):
            raise ValueError("require 0 < deviation < 0.45 * IQ sample rate")
        self.reset()

    def reset(self):
        self.phase = self.initial_phase % (2*np.pi)

    def process(self, message):
        m = _vector(message)
        if not len(m):
            return np.empty(0, dtype=np.complex128)
        if np.any(np.abs(m) > 1.0 + 1e-12):
            raise ValueError("normalized message exceeds [-1, 1]")
        phase = self.phase + np.cumsum(2*np.pi*self.freq_deviation*m/self.sample_rate)
        self.phase = float(phase[-1] % (2*np.pi))
        return np.exp(1j*phase)


class FMDiscriminator:
    """Recover normalized message from complex IQ phase increments.

    ``product`` and ``unwrap`` are algebraically equivalent while every
    increment is strictly within (-pi, pi).  They are comparisons of two
    implementations, not distinct FM inventions.  The first sample of an
    unknown external capture is set to zero because its prior phase is absent.
    """

    def __init__(self, sample_rate, freq_deviation, *, method="product",
                 frequency_offset_hz=0.0, min_magnitude=1e-8):
        self.sample_rate = float(sample_rate)
        self.freq_deviation = float(freq_deviation)
        self.frequency_offset_hz = float(frequency_offset_hz)
        self.min_magnitude = float(min_magnitude)
        self.method = method
        if not all(map(np.isfinite, (self.sample_rate, self.freq_deviation,
                                      self.frequency_offset_hz, self.min_magnitude))):
            raise ValueError("parameters must be finite")
        if not (self.sample_rate > 0 and 0 < self.freq_deviation < 0.45*self.sample_rate):
            raise ValueError("invalid deviation or IQ sample rate")
        if self.method not in ("product", "unwrap") or self.min_magnitude < 0:
            raise ValueError("invalid discriminator method or minimum magnitude")
        self.reset()

    def reset(self):
        self.previous = None

    def process(self, iq):
        z = _vector(iq, complex_input=True)
        if not len(z):
            return np.empty(0, dtype=np.float64)
        first_block = self.previous is None
        previous = z[0] if first_block else self.previous
        if self.method == "product":
            prev = np.concatenate(([previous], z[:-1]))
            delta = np.angle(z * np.conj(prev))
            valid = (np.abs(z) >= self.min_magnitude) & (np.abs(prev) >= self.min_magnitude)
        else:
            phase = np.unwrap(np.angle(np.concatenate(([previous], z))))
            delta = np.diff(phase)
            valid = np.abs(z) >= self.min_magnitude
            valid[0] &= abs(previous) >= self.min_magnitude
            if len(z) > 1:
                valid[1:] &= np.abs(z[:-1]) >= self.min_magnitude
        self.previous = z[-1]
        out = (delta*self.sample_rate/(2*np.pi) - self.frequency_offset_hz) / self.freq_deviation
        out = np.where(valid, out, 0.0)
        if first_block:
            out[0] = 0.0
        return out


class IQChannel:
    """Optional amplitude, phase, CFO, low-pass bandwidth and complex AWGN.

    ``bandwidth_hz`` is the two-sided channel width; the low-pass cutoff is
    bandwidth_hz/2.  SNR is mean complex signal power divided by mean total
    complex noise power, measured over each supplied block.
    """

    def __init__(self, sample_rate, *, snr_db=None, cfo_hz=0.0,
                 initial_phase=0.0, bandwidth_hz=None, amplitude=1.0, seed=0):
        self.sample_rate = float(sample_rate)
        self.snr_db = None if snr_db is None else float(snr_db)
        self.cfo_hz = float(cfo_hz)
        self.initial_phase = float(initial_phase)
        self.amplitude = float(amplitude)
        self.bandwidth_hz = None if bandwidth_hz is None else float(bandwidth_hz)
        values = (self.sample_rate, self.cfo_hz, self.initial_phase, self.amplitude)
        if self.snr_db is not None:
            values += (self.snr_db,)
        if self.bandwidth_hz is not None:
            values += (self.bandwidth_hz,)
        if not all(map(np.isfinite, values)) or self.sample_rate <= 0 or self.amplitude <= 0:
            raise ValueError("invalid channel parameters")
        if self.bandwidth_hz is not None and not 0 < self.bandwidth_hz < 0.9*self.sample_rate:
            raise ValueError("bandwidth must be below 0.9 * IQ sample rate")
        self.sos = (None if self.bandwidth_hz is None else
                    butter(4, self.bandwidth_hz/2, fs=self.sample_rate,
                           btype="low", output="sos"))
        self.seed = seed
        self.reset()

    def reset(self):
        self.phase = self.initial_phase % (2*np.pi)
        self.filter_state = (None if self.sos is None else
                             np.zeros((len(self.sos), 2), dtype=np.complex128))
        self.rng = np.random.default_rng(self.seed)

    def process(self, iq):
        z = _vector(iq, complex_input=True)
        if not len(z):
            return z.copy()
        phase = self.phase + 2*np.pi*self.cfo_hz*np.arange(1, len(z)+1)/self.sample_rate
        self.phase = float(phase[-1] % (2*np.pi))
        out = self.amplitude*z*np.exp(1j*phase)
        if self.sos is not None:
            out, self.filter_state = sosfilt(self.sos, out, zi=self.filter_state)
        if self.snr_db is not None:
            out = awgn(out, self.snr_db, self.rng)
        return out


class IntegerInterpolator:
    """Causal FIR audio-to-IQ resampler for an integer rate factor."""

    def __init__(self, input_rate, factor):
        self.input_rate = int(input_rate)
        self.factor = int(factor)
        if self.input_rate <= 0 or self.factor < 1:
            raise ValueError("invalid rate or factor")
        self.output_rate = self.input_rate*self.factor
        self.taps = (firwin(10*self.factor+1, 0.45*self.input_rate,
                            fs=self.output_rate)*self.factor)
        self.reset()

    def reset(self):
        self.state = np.zeros(len(self.taps)-1)

    def process(self, audio):
        x = _vector(audio)
        if not len(x):
            return np.empty(0)
        up = np.zeros(len(x)*self.factor)
        up[::self.factor] = x
        out, self.state = lfilter(self.taps, [1.0], up, zi=self.state)
        return out


class IntegerDecimator:
    """Causal anti-alias FIR IQ-rate audio to an integer lower rate."""

    def __init__(self, input_rate, factor, cutoff_hz=None):
        self.input_rate = int(input_rate)
        self.factor = int(factor)
        if self.input_rate <= 0 or self.factor < 1 or self.input_rate % self.factor:
            raise ValueError("IQ rate must be an integer multiple of audio rate")
        self.output_rate = self.input_rate//self.factor
        cutoff = 0.45*self.output_rate if cutoff_hz is None else float(cutoff_hz)
        if not 0 < cutoff < 0.5*self.output_rate:
            raise ValueError("cutoff must be below output Nyquist")
        self.taps = firwin(10*self.factor+1, cutoff, fs=self.input_rate)
        self.reset()

    def reset(self):
        self.state = np.zeros(len(self.taps)-1)
        self.offset = 0

    def process(self, audio_at_iq_rate):
        x = _vector(audio_at_iq_rate)
        if not len(x):
            return np.empty(0)
        filtered, self.state = lfilter(self.taps, [1.0], x, zi=self.state)
        first = (-self.offset) % self.factor
        out = filtered[first::self.factor]
        self.offset = (self.offset + len(x)) % self.factor
        return out
