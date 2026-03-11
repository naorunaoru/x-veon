import torch, time

def stress_pcie(duration=300):
    dev = torch.device('cuda')
    n_streams = 8
    streams = [torch.cuda.Stream() for _ in range(n_streams)]

    # 512MB per buffer x 8 = 4GB per set, 2 sets = 8GB device, fits in 16GB
    buf_shape = (128, 1024, 1024)
    bufs_h = [torch.randn(buf_shape, pin_memory=True) for _ in range(n_streams)]
    bufs_d = [torch.empty(buf_shape, device=dev) for _ in range(n_streams)]
    bufs_h2 = [torch.randn(buf_shape, pin_memory=True) for _ in range(n_streams)]
    bufs_d2 = [torch.empty(buf_shape, device=dev) for _ in range(n_streams)]

    print(f'Stressing PCIe: {n_streams} streams, 2x{n_streams*0.5:.0f} GB on device')
    errors = 0
    iters = 0
    t0 = time.time()

    while time.time() - t0 < duration:
        for i in range(n_streams):
            with torch.cuda.stream(streams[i]):
                bufs_d[i].copy_(bufs_h[i], non_blocking=True)
                bufs_h2[i].copy_(bufs_d2[i], non_blocking=True)

        for i in range(n_streams):
            with torch.cuda.stream(streams[i]):
                bufs_d2[i].copy_(bufs_h2[i], non_blocking=True)
                bufs_h[i].copy_(bufs_d[i], non_blocking=True)

        torch.cuda.synchronize()

        if iters % 20 == 0:
            ref = torch.randn(2048, 2048, pin_memory=True)
            if not torch.equal(ref, ref.cuda().cpu()):
                errors += 1
                print(f'!! CORRUPTION iter {iters} !!')
            print(f'{time.time()-t0:.0f}s | iter {iters} | {errors} errors')
        iters += 1

    print(f'Done: {iters} iters, {errors} errors')

stress_pcie()
