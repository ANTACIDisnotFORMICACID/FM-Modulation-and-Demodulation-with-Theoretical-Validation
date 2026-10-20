"""Mono broadcast-FM receiver for complex IQ recordings."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

from .sdr import FMDiscriminator, IntegerDecimator, _vector


class BroadcastFMReceiver:
    """Tune, channel-filter, discriminate, de-emphasize and decimate WBFM.

    The IQ rate must be an integer multiple of the desired audio rate.  This
    keeps every filter and decimation phase continuous across arbitrary blocks.
    ``tune_offset_hz`` is the station's offset from the recording center; a
    positive value is mixed down by a negative complex exponential.
    """

    def __init__(self, iq_sample_rate, *, decimation=5,
                 tune_offset_hz=0.0, channel_bandwidth_hz=200000.0,
                 freq_deviation=75000.0, audio_cutoff_hz=15000.0,
                 deemphasis_us=75.0, audio_highpass_hz=25.0):
        self.iq_sample_rate = int(iq_sample_rate)
        self.decimation = int(decimation)
        self.audio_sample_rate = self.iq_sample_rate//self.decimation
        self.tune_offset_hz = float(tune_offset_hz)
        self.channel_bandwidth_hz = float(channel_bandwidth_hz)
        self.freq_deviation = float(freq_deviation)
        self.deemphasis_us = float(deemphasis_us)
        if self.iq_sample_rate <= 0 or self.decimation < 1 or self.iq_sample_rate % self.decimation:
            raise ValueError("IQ sample rate must divide evenly by decimation")
        vals = (self.tune_offset_hz, self.channel_bandwidth_hz,
                self.freq_deviation, self.deemphasis_us, audio_highpass_hz)
        if not all(map(np.isfinite, vals)):
            raise ValueError("receiver parameters must be finite")
        if not 0 < self.channel_bandwidth_hz < 0.9*self.iq_sample_rate:
            raise ValueError("channel width must be below 0.9 * IQ sample rate")
        if not 0 < audio_highpass_hz < audio_cutoff_hz < 0.5*self.audio_sample_rate:
            raise ValueError("audio cutoff must lie between high-pass and output Nyquist")
        if not 0 <= self.deemphasis_us <= 1000:
            raise ValueError("de-emphasis must be 0..1000 microseconds")
        self.channel_sos = butter(4, self.channel_bandwidth_hz/2,
                                  fs=self.iq_sample_rate, output="sos")
        self.highpass_sos = butter(2, audio_highpass_hz,
                                   fs=self.iq_sample_rate, btype="high", output="sos")
        self.discriminator = FMDiscriminator(self.iq_sample_rate, self.freq_deviation)
        self.decimator = IntegerDecimator(self.iq_sample_rate, self.decimation,
                                           cutoff_hz=audio_cutoff_hz)
        self.deemphasis_a = (0.0 if self.deemphasis_us == 0 else
                             np.exp(-1/(self.deemphasis_us*1e-6*self.iq_sample_rate)))
        self.reset()

    def reset(self):
        self.lo_phase = 0.0
        self.channel_state = np.zeros((len(self.channel_sos), 2), dtype=np.complex128)
        self.highpass_state = np.zeros((len(self.highpass_sos), 2), dtype=np.float64)
        self.deemphasis_state = np.zeros(1)
        self.discriminator.reset()
        self.decimator.reset()
        self.last_discriminator = np.empty(0)

    def process(self, iq):
        x = _vector(iq, complex_input=True)
        if not len(x):
            return np.empty(0)
        phase = self.lo_phase - 2*np.pi*self.tune_offset_hz*np.arange(1, len(x)+1)/self.iq_sample_rate
        self.lo_phase = float(phase[-1] % (2*np.pi))
        mixed = x*np.exp(1j*phase)
        selected, self.channel_state = sosfilt(self.channel_sos, mixed,
                                                 zi=self.channel_state)
        demod = self.discriminator.process(selected)
        self.last_discriminator = demod
        audio, self.highpass_state = sosfilt(self.highpass_sos, demod,
                                              zi=self.highpass_state)
        a = self.deemphasis_a
        if a:
            audio, self.deemphasis_state = lfilter([1-a], [1, -a], audio,
                                                    zi=self.deemphasis_state)
        return self.decimator.process(audio)
