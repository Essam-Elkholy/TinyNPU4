"""Image preprocessing and host command schedule for CatDog TinyNPU."""

import json
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image, ImageOps

CMD_NOP = 0
CMD_CLEAR = 1
CMD_LOAD_BIAS_LOW = 2
CMD_LOAD_BIAS_HIGH = 3
CMD_MAC = 4
CMD_FINISH_RELU = 5
CMD_FINISH_LINEAR = 6


class ChipTransport(Protocol):
    def command(self, cmd: int, param: int = 0, data: int = 0) -> int:
        """Send one command pulse and return uo_out as an unsigned byte."""


def int4_nibble(value):
    value = int(value)
    if not -8 <= value <= 7:
        raise ValueError(f"INT4 value out of range: {value}")
    return value & 0xF


def preprocess_image(path, input_scale=0.125):
    image = Image.open(path).convert("L")
    image = ImageOps.pad(image, (8, 8), method=Image.Resampling.BILINEAR, color=0)
    normalized = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return np.clip(np.rint(normalized / input_scale), -8, 7).astype(np.int8).reshape(-1)


def load_bias(chip, bias, lane=0):
    value = int(bias) & 0xFF
    chip.command(CMD_LOAD_BIAS_LOW, value & 0xF, lane & 1)
    chip.command(CMD_LOAD_BIAS_HIGH, (value >> 4) & 0xF, lane & 1)


def run_neuron(chip, activations, weights, bias, shift, relu):
    if len(activations) != len(weights):
        raise ValueError("Activation and weight counts differ")
    chip.command(CMD_CLEAR)
    load_bias(chip, bias, lane=0)
    for activation, weight in zip(activations, weights):
        data = (int4_nibble(weight) << 4) | int4_nibble(activation)
        # Lane 1 receives a zero weight while lane 0 evaluates one neuron.
        chip.command(CMD_MAC, 0, data)
    command = CMD_FINISH_RELU if relu else CMD_FINISH_LINEAR
    status = chip.command(command, int(shift), 0)
    if not ((status >> 7) & 1):
        raise RuntimeError("Chip did not assert result_valid")
    result = status & 0xF
    signed_result = result - 16 if result & 8 else result
    return signed_result, (status >> 6) & 1


def run_neuron_pair(chip, activations, weights_0, bias_0,
                    weights_1, bias_1, shift, relu):
    """Evaluate two neurons in parallel from one activation stream."""
    if len(activations) != len(weights_0) or len(activations) != len(weights_1):
        raise ValueError("Activation and weight counts differ")

    chip.command(CMD_CLEAR)
    load_bias(chip, bias_0, lane=0)
    load_bias(chip, bias_1, lane=1)

    for activation, weight_0, weight_1 in zip(activations, weights_0, weights_1):
        data = (int4_nibble(weight_0) << 4) | int4_nibble(activation)
        chip.command(CMD_MAC, int4_nibble(weight_1), data)

    command = CMD_FINISH_RELU if relu else CMD_FINISH_LINEAR
    results = []
    for lane in (0, 1):
        status = chip.command(command, int(shift), lane)
        if not ((status >> 7) & 1):
            raise RuntimeError("Chip did not assert result_valid")
        result = status & 0xF
        results.append(result - 16 if result & 8 else result)
    return tuple(results)


def infer_on_chip(chip, model, pixels):
    hidden = []
    hidden_weights = model["layer1"]["weights"]
    hidden_biases = model["layer1"]["bias"]
    for index in range(0, len(hidden_weights), 2):
        weights_0 = hidden_weights[index]
        bias_0 = hidden_biases[index]
        if index + 1 < len(hidden_weights):
            weights_1 = hidden_weights[index + 1]
            bias_1 = hidden_biases[index + 1]
        else:
            weights_1 = [0] * len(pixels)
            bias_1 = 0

        value_0, value_1 = run_neuron_pair(
            chip, pixels, weights_0, bias_0, weights_1, bias_1,
            model["layer1"]["shift"], True
        )
        hidden.append(value_0)
        if index + 1 < len(hidden_weights):
            hidden.append(value_1)

    score, negative = run_neuron(
        chip, hidden, model["layer2"]["weights"][0],
        model["layer2"]["bias"][0], model["layer2"]["shift"], False
    )
    return {"score_int4": score, "class_id": 0 if negative else 1,
            "label": "dog" if negative else "cat", "hidden": hidden}


def load_model(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# Implement ChipTransport.command() for the exact Tiny Tapeout demo-board API,
# then call preprocess_image(), load_model() and infer_on_chip().
