## TinyNPU 64-4-1

[![GDS](../../actions/workflows/gds.yaml/badge.svg)](../../actions/workflows/gds.yaml)
[![Docs](../../actions/workflows/docs.yaml/badge.svg)](../../actions/workflows/docs.yaml)
[![Test](../../actions/workflows/test.yaml/badge.svg)](../../actions/workflows/test.yaml)

A Tiny Tapeout INT4 neural-network accelerator that classifies an 8x8 grayscale
image as **Cat** or **Dog**. Training and quantization happen on the host CPU;
the chip performs the actual hidden-layer and output-layer arithmetic on a
command-driven, bit-serial datapath.

## How it works

CatDog TinyNPU is a host-controlled **64-4-1** quantized neural-network
accelerator:

- **64** signed INT4 input pixels (an 8x8 grayscale image, quantized off-chip)
- **4** hidden neurons with ReLU activation
- **1** output neuron with Linear activation

The host streams pixels and weights over the pin interface. Two signed INT4
MAC lanes share each streamed pixel but consume two independent weights, so
two hidden neurons are computed in parallel — cutting the hidden layer from
256 MAC clocks down to 128. Each lane has its own signed 16-bit saturating
accumulator. `ui_in[0]` selects a lane during bias-load, finish, and debug
commands.

After the output neuron finishes, `uo_out[6] = 0` means **Cat** and
`uo_out[6] = 1` means **Dog**; `uo_out[3:0]` holds the saturated signed INT4
score.

## Architecture

```text
64 signed INT4 pixels
        |
        v
4 hidden ReLU neurons: two computed in parallel, 64 clocks per pair
        |
        v
1 Linear output neuron: 4 MACs
        |
        v
score >= 0: Cat   score < 0: Dog
```

The design contains two physical signed INT4 multipliers. They share each
input activation and apply two independent weights, so two hidden neurons
accumulate in parallel. The network still performs 260 mathematical MAC
operations in total, but the hidden layer only takes 128 MAC command clocks
instead of 256. Biases are signed INT8; each lane keeps an independent signed
16-bit saturating accumulator with overflow detection.

## Repository structure

```text
src/tt_um_tinynpu4.v          Tiny Tapeout top module: tt_um_tinynpu4
src/tinynpu4_core.v           Command controller and accumulator
src/int4_mac.sv               Signed INT4 multiplier
src/activation_quantizer.sv   ReLU/Linear, shift and saturation
src/config.json                Design configuration

test/tb_tt_um_tinynpu4.v      Cocotb wrapper
test/test.py                   RTL and overflow tests
test/tb.gtkw                   GTKWave signal layout

ml/train_and_quantize.py       Cat/Dog training and INT4 export
ml/chip_inference.py           Host-to-chip command sequence
ml/artifacts/model_int4.json   Generated weights, biases and shifts (after training)

docs/info.md                   Tiny Tapeout project documentation
info.yaml                       Tiny Tapeout project/pinout metadata
```

## Pins

```text
ui_in[0]     activation[0] / lane select (bias, finish, debug commands)
ui_in[3:1]   activation[3:1]
ui_in[7:4]   lane 0 weight during MAC

uio_in[2:0]  command
uio_in[3]    command_valid
uio_in[7:4]  lane 1 weight during MAC; otherwise bias nibble, shift, or debug index

uo_out[3:0]  signed INT4 result
uo_out[4]    done
uo_out[5]    overflow
uo_out[6]    final accumulator sign: 0 = Cat, 1 = Dog
uo_out[7]    result_valid
```

For bias-load, finish, and accumulator-read commands, `ui_in[0]` selects lane 0
or lane 1. Command `111` reads that lane's accumulator one nibble at a time;
`uio_in[5:4]` selects nibble 0, 1, 2, or 3.

## How to test

The automated Cocotb testbench lives in `test/test.py`:

```sh
cd test
python -m pip install -r requirements.txt
make
```

The test checks reset behavior, pin directions, a complete two-lane 64-4-1
inference schedule, both MAC lanes, ReLU, Linear saturation, the final class
sign, independent signed 16-bit saturation boundaries, and full accumulator
debug readback.

## Train the model

Train and quantize the real Cat/Dog model separately from RTL simulation:

```sh
python -m pip install -r ml/requirements.txt
python ml/train_and_quantize.py
```

This produces `ml/artifacts/model_int4.json`, containing the weights, biases,
and shifts that get streamed to the chip. Cat/Dog accuracy is only meaningful
after this training and integer-quantized evaluation step completes.

## External hardware

No external hardware is needed for RTL simulation. Running on a fabricated
chip requires a Tiny Tapeout demo board and a host computer or microcontroller
to drive the input and command pins — no PMOD or display is required.

## Project info

| | |
|---|---|
| **Title** | CatDog TinyNPU 64-4-1 |
| **Author** | Essam Elkholy and Team |
| **Language** | SystemVerilog |
| **Clock** | 50 MHz |
| **Tiles** | 1x1 |
| **Top module** | `tt_um_tinynpu4` |

See [`docs/info.md`](docs/info.md) for the full command protocol and hardware
usage details.
