## How it works

TinyNPU4 (also referred to as CatDog TinyNPU) is a host-controlled 64-4-1
quantized neural-network accelerator. The host converts a photo to 8x8
grayscale signed INT4 pixels. Two signed INT4 MAC lanes share each streamed
pixel and consume two independent weights, so two hidden neurons are computed
in parallel. Each lane has a signed 16-bit saturating accumulator. Hidden
results use ReLU; the final result uses Linear activation. `ui_in[0]` selects
a lane during bias, finish and debug commands.

After the output neuron, `uo_out[6] = 0` means Cat and `uo_out[6] = 1` means
Dog. `uo_out[3:0]` is the saturated signed INT4 score.

## How to test

The automated Cocotb test is in `test/test.py`. From the `test` directory run:

```sh
make
```

The test checks reset, pin directions, a complete two-lane 64-4-1 inference
schedule, both MAC lanes, ReLU, Linear saturation, final class sign, independent
signed 16-bit saturation boundaries, and full accumulator debug readback.

A separate ModelSim demo (`run_trained_demo.do`, driving
`tb_catdog_trained_model.sv`) replays the trained and quantized model against
10 real cat/dog photos directly through the RTL. All 10 were classified
correctly (5/5 cats, 5/5 dogs).

Train and quantize the real Cat/Dog model separately with:

```sh
python -m pip install -r ml/requirements.txt
python ml/train_and_quantize.py
```

Training creates `ml/artifacts/model_int4.json`, containing the weights, biases
and shifts streamed to the chip.

## External hardware

No external hardware is needed for RTL simulation. A fabricated-chip test needs
a Tiny Tapeout demo board and a host computer or microcontroller to drive the
input and command pins. No PMOD or display is required.
