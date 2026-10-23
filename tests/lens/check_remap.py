#!/usr/bin/env python3
"""Run the production WGSL lens remap on WebGPU and compare with its CPU twin.

Requires numpy, wgpu, Node and installed web dependencies. remap_cases.ts builds
three remaps with the production table builder and uniform packing, and computes
the expected sources with lensSourceUv. The compute entry point appended to the
production shader calls lens_remap() unchanged: texture sampling is not part of
the comparison. Uniforms are bound at 0 and the radial table at 5, as in the
renderer; the uvs and results are test buffers at 3 and 4.
"""
import json
import pathlib
import subprocess
import tempfile

import numpy as np
import wgpu

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
UV_TOLERANCE = 1e-5  # about 0.03 px on a 3000 px frame


def run():
    with tempfile.TemporaryDirectory(prefix='lens-test-') as directory:
        bundle = pathlib.Path(directory) / 'cases.mjs'
        subprocess.run([str(ROOT / 'node_modules/.bin/esbuild'), str(HERE / 'remap_cases.ts'), '--bundle', '--platform=node', '--format=esm', f'--outfile={bundle}'], check=True)
        cases = json.loads(subprocess.check_output(['node', str(bundle)]))
    adapter = wgpu.gpu.request_adapter_sync()
    print('GPU:', dict(adapter.info))
    device = adapter.request_device_sync()
    source = (ROOT / 'shared/src/renderer/shaders/opendrt.wgsl').read_text() + '''
@group(0) @binding(3) var<storage,read> test_in: array<vec2f>;
@group(0) @binding(4) var<storage,read_write> test_out: array<vec4f>;
@compute @workgroup_size(64) fn test_main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x < arrayLength(&test_in)) {
    let s = lens_remap(test_in[id.x]);
    test_out[2u * id.x] = vec4f(s.uv_r, s.uv_g);
    test_out[2u * id.x + 1u] = vec4f(s.uv_b, s.gain, 0.0);
  }
}
'''
    pipeline = device.create_compute_pipeline(layout='auto', compute={'module': device.create_shader_module(code=source), 'entry_point': 'test_main'})
    worst = 0.0
    worst_gain = 0.0
    points = 0
    buffers = []
    for case in cases:
        name = case['name']
        uvs = np.array(case['uvs'], np.float32)
        expected = np.array([[*e['r'], *e['g'], *e['b'], e['gain'], 0] for e in case['expected']], np.float32)
        ub = device.create_buffer_with_data(data=np.array(case['uniforms'], np.float32), usage=wgpu.BufferUsage.UNIFORM)
        tb = device.create_buffer_with_data(data=np.array(case['table'], np.float32), usage=wgpu.BufferUsage.STORAGE)
        ib = device.create_buffer_with_data(data=uvs, usage=wgpu.BufferUsage.STORAGE)
        ob = device.create_buffer(size=expected.nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        buffers += [ub, tb, ib, ob]
        group = device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=[{'binding': i, 'resource': {'buffer': b}} for i, b in [(0, ub), (3, ib), (4, ob), (5, tb)]])
        encoder = device.create_command_encoder()
        compute = encoder.begin_compute_pass()
        compute.set_pipeline(pipeline)
        compute.set_bind_group(0, group)
        compute.dispatch_workgroups((len(uvs) + 63) // 64)
        compute.end()
        device.queue.submit([encoder.finish()])
        out = np.frombuffer(device.queue.read_buffer(ob), np.float32).reshape(-1, 8)
        assert out.shape == expected.shape, f'{name}: {out.shape[0]} results for {len(uvs)} points'
        assert np.isfinite(out).all(), f'{name}: nonfinite output'
        for i, uv in enumerate(uvs):
            error = float(np.max(np.abs(out[i, :6] - expected[i, :6])))
            gain_error = float(abs(out[i, 6] - expected[i, 6]))
            assert error < UV_TOLERANCE and gain_error < UV_TOLERANCE, \
                f'{name}: point {i} at uv {uv.tolist()}: GPU {out[i, :7].tolist()} vs CPU {expected[i, :7].tolist()} (uv error {error:.3g}, gain error {gain_error:.3g})'
            worst = max(worst, error)
            worst_gain = max(worst_gain, gain_error)
        points = len(uvs)
    print(f'maximum gain error {worst_gain:.8g}')
    print(f'PASS: {len(cases)} remaps × {points} points; maximum uv error {worst:.8g}')
    for buffer in buffers:
        buffer.destroy()
    device.destroy()


if __name__ == '__main__':
    run()
