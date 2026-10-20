"""Command line entry points for the software FM communication system."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .iqio import FileIQSink, FileIQSource
from .receiver import BroadcastFMReceiver
from .sdr import (ComplexFMModulator, FMDiscriminator, IQChannel,
                  IntegerDecimator, IntegerInterpolator)


def _save_json(path, data):
    if path is not None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")


def simulate(args):
    import soundfile as sf

    output_path = Path(args.output).resolve()
    iq_path = None if args.iq_out is None else Path(args.iq_out).resolve()
    if iq_path == output_path:
        raise ValueError("IQ and WAV output paths must differ")
    if args.input is not None and iq_path == Path(args.input).resolve():
        raise ValueError("IQ output must differ from input WAV")
    source = None
    if args.input is not None:
        info = sf.info(args.input)
        if info.format != "WAV" or info.channels not in (1, 2):
            raise ValueError("input must be mono or stereo WAV")
        if Path(args.input).resolve() == Path(args.output).resolve():
            raise ValueError("input and output must differ")
        sample_rate = info.samplerate
        source = sf.SoundFile(args.input)
        total = info.frames
    else:
        sample_rate = args.audio_rate
        total = int(round(args.duration*sample_rate))
        if total <= 0:
            raise ValueError("duration must produce at least one sample")
    iq_rate = sample_rate*args.rate_factor
    interp = IntegerInterpolator(sample_rate, args.rate_factor)
    mod = ComplexFMModulator(iq_rate, args.freq_deviation)
    channel = IQChannel(iq_rate, snr_db=args.snr_db, cfo_hz=args.cfo_hz,
                        initial_phase=args.initial_phase,
                        bandwidth_hz=args.bandwidth_hz,
                        amplitude=args.amplitude, seed=args.seed)
    demod = FMDiscriminator(iq_rate, args.freq_deviation,
                            method=args.method,
                            frequency_offset_hz=(args.cfo_hz if args.correct_cfo else 0))
    decim = IntegerDecimator(iq_rate, args.rate_factor,
                              cutoff_hz=min(args.audio_cutoff, 0.45*sample_rate))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    iq_sink = (None if args.iq_out is None else
               FileIQSink(args.iq_out, sample_rate=iq_rate, source="software FM simulation"))
    input_sum = output_sum = 0.0
    input_peak = output_peak = 0.0
    frames = 0
    try:
        with sf.SoundFile(output, "w", samplerate=sample_rate,
                          channels=1, format="WAV", subtype="PCM_24") as wav:
            for pos in range(0, total, args.block_size):
                n = min(args.block_size, total-pos)
                if source is None:
                    t = np.arange(pos, pos+n)/sample_rate
                    audio = 0.5*np.sin(2*np.pi*1000*t) + 0.2*np.sin(2*np.pi*2000*t)
                else:
                    audio = source.read(n, dtype="float64", always_2d=True).mean(axis=1)
                if not np.all(np.isfinite(audio)) or np.any(np.abs(audio) > 1):
                    raise ValueError("audio must be finite and within [-1, 1]")
                iq = mod.process(interp.process(audio))
                if iq_sink is not None:
                    iq_sink.write(iq)
                received = channel.process(iq)
                restored = decim.process(demod.process(received))
                if np.max(np.abs(restored)) > 1:
                    raise ValueError("recovered audio exceeds PCM range; adjust channel/gain")
                wav.write(restored)
                input_sum += float(np.sum(audio**2))
                output_sum += float(np.sum(restored**2))
                input_peak = max(input_peak, float(np.max(np.abs(audio))))
                output_peak = max(output_peak, float(np.max(np.abs(restored))))
                frames += len(audio)
    finally:
        if source is not None:
            source.close()
        if iq_sink is not None:
            iq_sink.close()
    report = {"mode": "simulate", "input": args.input or "synthetic two-tone",
              "output": str(output), "iq_out": args.iq_out,
              "audio_sample_rate": sample_rate, "iq_sample_rate": iq_rate,
              "freq_deviation_hz": args.freq_deviation, "channel_snr_db": args.snr_db,
              "channel_cfo_hz": args.cfo_hz, "seed": args.seed,
              "frames": frames, "input_rms": np.sqrt(input_sum/frames),
              "input_peak": input_peak, "output_rms": np.sqrt(output_sum/frames),
              "output_peak": output_peak,
              "filter_note": "causal FIR interpolation and decimation introduce deterministic delay"}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    _save_json(args.report, report)


def receive(args):
    import soundfile as sf

    if Path(args.input).resolve() == Path(args.output).resolve():
        raise ValueError("IQ input and WAV output paths must differ")
    with FileIQSource(args.input, sample_rate=args.iq_rate,
                      fmt=args.format, center_freq=args.center_freq) as source:
        receiver = BroadcastFMReceiver(
            source.sample_rate, decimation=args.decimation,
            tune_offset_hz=args.tune_offset,
            channel_bandwidth_hz=args.channel_bandwidth,
            freq_deviation=args.freq_deviation,
            audio_cutoff_hz=args.audio_cutoff,
            deemphasis_us=args.deemphasis_us)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        samples = 0
        peak = power = 0.0
        first_iq = None
        first_demod = None
        first_audio = None
        with sf.SoundFile(output, "w", samplerate=receiver.audio_sample_rate,
                          channels=1, format="WAV", subtype="PCM_24") as wav:
            while True:
                iq = source.read(args.block_size)
                if not len(iq):
                    break
                audio = receiver.process(iq)*args.audio_gain
                if np.any(np.abs(audio) > 1):
                    raise ValueError("audio exceeds PCM range; reduce --audio-gain")
                wav.write(audio)
                if first_iq is None:
                    first_iq = iq[:min(len(iq), 65536)].copy()
                    first_demod = receiver.last_discriminator[:min(len(iq), 65536)].copy()
                    first_audio = audio[:min(len(audio), 10000)].copy()
                peak = max(peak, float(np.max(np.abs(audio))))
                power += float(np.sum(audio*audio))
                samples += len(audio)
    if not samples:
        raise ValueError("IQ file is empty")
    report = {"mode": "receive", "input": args.input, "output": str(output),
              "iq_format": source.fmt, "iq_sample_rate": source.sample_rate,
              "audio_sample_rate": receiver.audio_sample_rate,
              "center_freq_hz": source.center_freq,
              "tune_offset_hz": args.tune_offset,
              "channel_bandwidth_hz": args.channel_bandwidth,
              "freq_deviation_hz": args.freq_deviation,
              "deemphasis_us": args.deemphasis_us,
              "audio_gain": args.audio_gain, "audio_samples": samples,
              "audio_duration_s": samples/receiver.audio_sample_rate,
              "audio_peak": peak, "audio_rms": np.sqrt(power/samples)}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    _save_json(args.report, report)
    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.signal import welch

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
        f, p = welch(first_iq, fs=source.sample_rate,
                     nperseg=min(4096, len(first_iq)), return_onesided=False)
        order = np.argsort(f)
        axes[0].plot(f[order]/1000, 10*np.log10(np.maximum(p[order], 1e-20)))
        axes[0].set(xlabel="Offset frequency (kHz)", ylabel="PSD (dB/Hz)",
                    title="Recorded IQ spectrum")
        axes[1].plot(np.arange(len(first_demod))/source.sample_rate,
                     first_demod, linewidth=0.5)
        axes[1].set(xlabel="Time (s)", ylabel="Normalized deviation",
                    title="FM discriminator (first block)")
        axes[2].plot(np.arange(len(first_audio))/receiver.audio_sample_rate,
                     first_audio, linewidth=0.5)
        axes[2].set(xlabel="Time (s)", ylabel="Amplitude", title="Recovered mono audio")
        target = Path(args.figure)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=150)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m fm_project")
    sub = parser.add_subparsers(dest="command", required=True)
    sim = sub.add_parser("simulate", help="WAV or test signal through FM IQ channel")
    sim.add_argument("--input", help="mono or stereo WAV (stereo downmixed)")
    sim.add_argument("--output", required=True)
    sim.add_argument("--iq-out")
    sim.add_argument("--report")
    sim.add_argument("--audio-rate", type=int, default=48000)
    sim.add_argument("--duration", type=float, default=1.0)
    sim.add_argument("--rate-factor", type=int, default=5)
    sim.add_argument("--freq-deviation", type=float, default=75000)
    sim.add_argument("--audio-cutoff", type=float, default=15000)
    sim.add_argument("--snr-db", type=float)
    sim.add_argument("--cfo-hz", type=float, default=0)
    sim.add_argument("--correct-cfo", action="store_true")
    sim.add_argument("--initial-phase", type=float, default=0)
    sim.add_argument("--bandwidth-hz", type=float)
    sim.add_argument("--amplitude", type=float, default=1)
    sim.add_argument("--seed", type=int, default=0)
    sim.add_argument("--method", choices=("product", "unwrap"), default="product")
    sim.add_argument("--block-size", type=int, default=4096)
    sim.set_defaults(action=simulate)
    rx = sub.add_parser("receive", help="mono FM audio from recorded complex IQ")
    rx.add_argument("--input", required=True)
    rx.add_argument("--output", required=True)
    rx.add_argument("--iq-rate", type=int, help="required if no IQ JSON metadata")
    rx.add_argument("--format", choices=("cf32le", "cs8", "ci16le"))
    rx.add_argument("--center-freq", type=float)
    rx.add_argument("--tune-offset", type=float, default=0)
    rx.add_argument("--decimation", type=int, default=5)
    rx.add_argument("--channel-bandwidth", type=float, default=200000)
    rx.add_argument("--freq-deviation", type=float, default=75000)
    rx.add_argument("--audio-cutoff", type=float, default=15000)
    rx.add_argument("--deemphasis-us", type=float, default=75)
    rx.add_argument("--audio-gain", type=float, default=0.5)
    rx.add_argument("--block-size", type=int, default=65536)
    rx.add_argument("--report")
    rx.add_argument("--figure")
    rx.set_defaults(action=receive)
    args = parser.parse_args(argv)
    if args.block_size <= 0 or (hasattr(args, "audio_gain") and
                                (not np.isfinite(args.audio_gain) or args.audio_gain <= 0)):
        parser.error("block size and audio gain must be positive")
    try:
        args.action(args)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
