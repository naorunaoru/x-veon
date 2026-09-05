#!/usr/bin/env python3
"""Run the production WGSL core on WebGPU and compare with the complete ART CTL.

Requires numpy, wgpu, Node, installed web dependencies and a C++ compiler.
No downloaded reference or hand-maintained CPU translation is used: compile the
bundled CTL transform as C++ (its scalar/vector helpers are C++ compatible).
Only rename three CTL helpers to avoid collisions with C++ standard functions.
The compute entry point calls opendrt() unchanged; texture sampling/OETF are not
part of the numerical comparison. CPU and GPU both receive the same P3 inputs.
"""
import ctypes
import json
import pathlib
import re
import subprocess
import tempfile

import numpy as np
import wgpu

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent


def reference(tmp):
    source = (HERE / 'opendrt_art.ctl').read_text().split('void ART_main')[0]
    for name in ['fmin', 'fmax', 'log2']:
        source = re.sub(r'\b' + name + r'\b', 'ctl_' + name, source)
    signature = re.search(r'float3 transform\((.*?)\)\s*\{', source, re.S).group(1)
    names = re.findall(r'(?:int|float)\s+(\w+)', signature)
    call = ','.join(f'args[{i}]' for i in range(len(names)))
    wrapper = '\nextern "C" void evaluate(float *args, float *out) { auto v = transform(' + call + '); out[0]=v.x; out[1]=v.y; out[2]=v.z; }\n'
    cpp = tmp / 'reference.cpp'
    cpp.write_text('#include <cmath>\n' + source + wrapper)
    library = tmp / 'reference.so'
    subprocess.run(['c++', '-std=c++17', '-shared', '-fPIC', '-O2', str(cpp), '-o', str(library)], check=True)
    lib = ctypes.CDLL(str(library))
    ptr = ctypes.POINTER(ctypes.c_float)
    lib.evaluate.argtypes = [ptr, ptr]

    def evaluate(case, inputs):
        cfg = case['cfg'].copy()
        cfg.update(in_gamut=3, eotf=0, display_gamut={'rec709': 0, 'p3': 1, 'rec2020': 2}[case['gamut']],
                   tn_Lp=cfg['peak_luminance'], tn_Lg=cfg['tn_lg'], tn_gb=cfg['grey_boost'], cwp=3 if cfg['cwp'] else 0)
        args = np.array([cfg.get(k, 0) for k in names], np.float32)
        out = np.empty(3, np.float32)
        matrix = np.array(case['uniforms'][52:64], np.float32).reshape(3, 4)[:, :3].T
        expected = []
        for rgb in inputs[:, :3]:
            args[:3] = matrix @ rgb
            lib.evaluate(args.ctypes.data_as(ptr), out.ctypes.data_as(ptr))
            expected.append(out.copy())
        return np.array(expected)
    return evaluate


def run():
    with tempfile.TemporaryDirectory(prefix='opendrt-test-') as directory:
        tmp = pathlib.Path(directory)
        bundle = tmp / 'cases.mjs'
        subprocess.run([str(ROOT / 'web/node_modules/.bin/esbuild'), str(HERE / 'shader_cases.ts'), '--bundle', '--platform=node', '--format=esm', f'--outfile={bundle}'], check=True)
        cases = json.loads(subprocess.check_output(['node', str(bundle)]))
        cpu = reference(tmp)
        adapter = wgpu.gpu.request_adapter_sync()
        print('GPU:', dict(adapter.info))
        device = adapter.request_device_sync()
        source = (ROOT / 'web/src/renderer/shaders/opendrt.wgsl').read_text() + '''
@group(0) @binding(3) var<storage,read> test_in: array<vec4f>;
@group(0) @binding(4) var<storage,read_write> test_out: array<vec4f>;
@compute @workgroup_size(64) fn test_main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x < arrayLength(&test_in)) { test_out[id.x] = vec4f(opendrt(test_in[id.x].xyz), 1.0); }
}
'''
        pipeline = device.create_compute_pipeline(layout='auto', compute={'module': device.create_shader_module(code=source), 'entry_point': 'test_main'})
        rng = np.random.default_rng(904)
        inputs = np.array([[v, v, v, 1] for v in [0, .001, .01, .18, 1, 10, 100]] + [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]], np.float32)
        inputs = np.concatenate([inputs, np.column_stack([10 ** rng.uniform(-4, 2, (4000, 3)), np.ones(4000)]).astype(np.float32)])
        ib = device.create_buffer_with_data(data=inputs, usage=wgpu.BufferUsage.STORAGE)
        ub = device.create_buffer(size=400, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        ob = device.create_buffer(size=inputs.nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        group = device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=[{'binding': i, 'resource': {'buffer': b}} for i, b in [(0, ub), (3, ib), (4, ob)]])
        worst = 0.0
        compared = 0
        warmth = {}
        for case in cases:
            device.queue.write_buffer(ub, 0, np.array(case['uniforms'], np.float32))
            encoder = device.create_command_encoder()
            compute = encoder.begin_compute_pass()
            compute.set_pipeline(pipeline)
            compute.set_bind_group(0, group)
            compute.dispatch_workgroups((len(inputs) + 63) // 64)
            compute.end()
            device.queue.submit([encoder.finish()])
            out = np.frombuffer(device.queue.read_buffer(ob), np.float32).reshape(-1, 4)[:, :3].copy()
            assert np.isfinite(out).all(), f"Nonfinite output: {case['name']}"
            assert np.max(out[3]) > 0, f"Middle grey became black: {case['name']}"
            if case['compare']:
                expected = cpu(case, inputs)
                error = float(np.max(np.abs(out - expected)))
                assert np.isfinite(expected).all() and error < 1e-5, f"Reference mismatch: {case['name']}: {error}"
                worst = max(worst, error)
                compared += 1
            if case['name'].startswith('warmth/'):
                warmth[case['cfg']['cwp']] = out[3]
        assert warmth[0][2] > warmth[.01][2] > warmth[.5][2] > warmth[1][2], 'Warmth must vary continuously'
        print(f'PASS: {len(cases)} configurations × {len(inputs)} colors; {compared} full CTL comparisons; maximum absolute error {worst:.8g}')
        for buffer in [ib, ub, ob]:
            buffer.destroy()
        device.destroy()


if __name__ == '__main__':
    run()
