"""Cocotb tests for the CatDog TinyNPU Tiny Tapeout interface."""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer

CMD_NOP = 0b000
CMD_CLEAR = 0b001
CMD_LOAD_BIAS_LOW = 0b010
CMD_LOAD_BIAS_HIGH = 0b011
CMD_MAC = 0b100
CMD_FINISH_RELU = 0b101
CMD_FINISH_LINEAR = 0b110
CMD_READ_ACC_NIBBLE = 0b111


def nibble(value):
    return int(value) & 0xF


async def send_command(dut, command, parameter=0, data=0):
    await FallingEdge(dut.clk)
    dut.ui_in.value = data & 0xFF
    dut.uio_in.value = ((parameter & 0xF) << 4) | (1 << 3) | command
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    status = int(dut.uo_out.value)
    await FallingEdge(dut.clk)
    dut.uio_in.value = CMD_NOP
    return status


async def clear_neuron(dut):
    await send_command(dut, CMD_CLEAR)


async def load_bias(dut, bias, lane=0):
    value = int(bias) & 0xFF
    await send_command(dut, CMD_LOAD_BIAS_LOW, value & 0xF, lane & 1)
    await send_command(dut, CMD_LOAD_BIAS_HIGH, (value >> 4) & 0xF, lane & 1)


async def dual_mac(dut, activation, weight_0, weight_1):
    data = (nibble(weight_0) << 4) | nibble(activation)
    await send_command(dut, CMD_MAC, parameter=nibble(weight_1), data=data)


async def run_neuron(dut, activations, weights, bias, shift, relu):
    assert len(activations) == len(weights)
    await clear_neuron(dut)
    await load_bias(dut, bias, lane=0)
    for activation, weight in zip(activations, weights):
        await dual_mac(dut, activation, weight, 0)
    command = CMD_FINISH_RELU if relu else CMD_FINISH_LINEAR
    status = await send_command(dut, command, parameter=shift, data=0)
    assert (status >> 4) & 1, "done was not asserted"
    assert (status >> 7) & 1, "result_valid was not asserted"
    result = status & 0xF
    return result - 16 if result & 8 else result, (status >> 6) & 1


async def run_neuron_pair(dut, activations, weights_0, bias_0,
                          weights_1, bias_1, shift, relu):
    """Evaluate two neurons from the same activations in parallel."""
    assert len(activations) == len(weights_0) == len(weights_1)
    await clear_neuron(dut)
    await load_bias(dut, bias_0, lane=0)
    await load_bias(dut, bias_1, lane=1)
    for activation, weight_0, weight_1 in zip(activations, weights_0, weights_1):
        await dual_mac(dut, activation, weight_0, weight_1)

    command = CMD_FINISH_RELU if relu else CMD_FINISH_LINEAR
    results = []
    for lane in (0, 1):
        status = await send_command(dut, command, parameter=shift, data=lane)
        assert (status >> 4) & 1, "done was not asserted"
        assert (status >> 7) & 1, "result_valid was not asserted"
        result = status & 0xF
        results.append(result - 16 if result & 8 else result)
    return tuple(results)


async def read_accumulator(dut, lane=0):
    """Read all four nibbles of one signed 16-bit accumulator."""
    value = 0
    for nibble_index in range(4):
        status = await send_command(
            dut, CMD_READ_ACC_NIBBLE, parameter=nibble_index, data=lane & 1
        )
        assert (status >> 4) & 1
        assert (status >> 7) & 1
        value |= (status & 0xF) << (4 * nibble_index)
    return value - 65536 if value & 0x8000 else value


@cocotb.test()
async def test_catdog_tinynpu(dut):
    """Check two-lane 64-4-1 inference and signed 16-bit saturation."""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    assert int(dut.uo_out.value) == 0
    assert int(dut.uio_out.value) == 0
    assert int(dut.uio_oe.value) == 0

    pixels = [7] * 64
    h0, h1 = await run_neuron_pair(
        dut, pixels, [7] * 64, 127, [-8] * 64, -128, 9, True
    )
    h2, h3 = await run_neuron_pair(
        dut, pixels, [7] * 64, 127, [0] * 64, 4, 9, True
    )
    assert [h0, h1, h2, h3] == [6, 0, 6, 0]

    score, negative = await run_neuron(
        dut, [h0, h1, h2, h3], [1, -1, 1, -1], 0, 0, False
    )
    assert score == 7
    assert negative == 0  # non-negative final accumulator means Cat
    assert ((int(dut.uo_out.value) >> 5) & 1) == 0

    # +32767 is valid; the following +64 saturates and sets overflow.
    await clear_neuron(dut)
    await load_bias(dut, 127, lane=0)
    await load_bias(dut, 0, lane=1)
    for _ in range(510):
        await dual_mac(dut, -8, -8, 0)
    assert ((int(dut.uo_out.value) >> 5) & 1) == 0
    assert await read_accumulator(dut, lane=0) == 32767
    assert await read_accumulator(dut, lane=1) == 0
    await dual_mac(dut, -8, -8, 0)
    assert ((int(dut.uo_out.value) >> 5) & 1) == 1
    assert await read_accumulator(dut, lane=0) == 32767

    # Build exactly -32768 in lane 1, then verify negative saturation below it.
    await clear_neuron(dut)
    await load_bias(dut, 0, lane=0)
    await load_bias(dut, -128, lane=1)
    for _ in range(582):
        await dual_mac(dut, 7, 0, -8)
    await dual_mac(dut, 6, 0, -8)
    assert ((int(dut.uo_out.value) >> 5) & 1) == 0
    assert await read_accumulator(dut, lane=1) == -32768
    assert await read_accumulator(dut, lane=0) == 0
    await dual_mac(dut, 7, 0, -8)
    assert ((int(dut.uo_out.value) >> 5) & 1) == 1
    assert await read_accumulator(dut, lane=1) == -32768

    dut._log.info("PASS: two-MAC 64-4-1 inference and independent saturation")
