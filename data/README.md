# IQ data

The software transceiver uses `fm_project/iqio.py` to save little-endian `complex64` IQ (`cf32le`, 8 bytes per sample) with an adjacent `.iq.json` metadata file. It can also read interleaved signed 8-bit (`cs8`) and little-endian signed 16-bit (`ci16le`) IQ. Specify the format and IQ sample rate when reading an external recording; these formats are not interchangeable.

The metadata fields are `sample_rate`, `center_freq`, `fmt`, `sample_count`, and `source`. Raw IQ files under `real/` and `generated/` are excluded from Git. See [real/README.md](real/README.md) for the external recording's provenance and reproduction steps.

`../legacy/original.iq` is an early synthetic file whose length does not match the output of its accompanying generator script. It is useful only for checking file-format handling, not for independently validating this project's demodulator.
