# CatDog TinyNPU tests

The Tiny Tapeout testbench uses Cocotb:

```sh
python -m pip install -r requirements.txt
make
```

`tb_tt_um_tinynpu4.v` instantiates the real top module, `tt_um_tinynpu4`.
`test.py` drives the same pin-level command protocol used by a host computer or
microcontroller.

The test covers reset, pin direction, four 64-input hidden neurons, the
four-input output neuron, two parallel MAC lanes, ReLU and Linear behavior,
Cat/Dog class sign, full accumulator readback, and signed 16-bit saturation.

GitHub Actions runs this test after every push. Gate-level testing reuses the
same Cocotb test after the GDS flow generates `gate_level_netlist.v`.
