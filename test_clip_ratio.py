import torch
from cfa import make_cfa_mask, make_channel_masks, CFA_REGISTRY


def test_clip_ratio():
    H, W = 48, 48
    pattern = CFA_REGISTRY["xtrans"]
    cfa = make_cfa_mask(H, W, pattern)

    wb = torch.tensor([2.0, 1.0, 1.5])  # typical Fuji WB
    clip_scale = 1.0
    clip_levels = wb[cfa.long()].unsqueeze(0) * clip_scale  # (1, H, W)

    # 1. All zeros → clip_ratio = 0
    cfa_img = torch.zeros(1, H, W)
    raw_ratio = (cfa_img / (clip_levels + 1e-8)).clamp(0, 1)
    clip_ratio = ((raw_ratio - 0.5) * 2.0).clamp(0, 1)
    assert clip_ratio.max() == 0.0, "zeros should give clip_ratio=0"

    # 2. Exactly at clip level → clip_ratio = 1.0
    cfa_img = clip_levels.clone()
    raw_ratio = (cfa_img / (clip_levels + 1e-8)).clamp(0, 1)
    clip_ratio = ((raw_ratio - 0.5) * 2.0).clamp(0, 1)
    assert clip_ratio.min() > 0.99, f"at clip level, clip_ratio should be ~1.0, got {clip_ratio.min()}"

    # 3. At 50% of clip level → clip_ratio = 0
    cfa_img = clip_levels * 0.5
    raw_ratio = (cfa_img / (clip_levels + 1e-8)).clamp(0, 1)
    clip_ratio = ((raw_ratio - 0.5) * 2.0).clamp(0, 1)
    assert clip_ratio.max() < 1e-5, f"at 50% clip, clip_ratio should be 0, got {clip_ratio.max()}"

    # 4. At 75% → clip_ratio = 0.5
    cfa_img = clip_levels * 0.75
    raw_ratio = (cfa_img / (clip_levels + 1e-8)).clamp(0, 1)
    clip_ratio = ((raw_ratio - 0.5) * 2.0).clamp(0, 1)
    assert (clip_ratio - 0.5).abs().max() < 1e-5, "at 75% clip, clip_ratio should be 0.5"

    # 5. Above clip level → still 1.0 (clamped)
    cfa_img = clip_levels * 2.0
    raw_ratio = (cfa_img / (clip_levels + 1e-8)).clamp(0, 1)
    clip_ratio = ((raw_ratio - 0.5) * 2.0).clamp(0, 1)
    assert clip_ratio.min() > 0.99, "above clip should clamp to 1.0"

    # 6. Per-photosite clip levels match CFA pattern
    r_mask = (cfa == 0)
    g_mask = (cfa == 1)
    b_mask = (cfa == 2)
    assert (clip_levels[0][r_mask] == wb[0] * clip_scale).all(), "R photosites should clip at wb[0]"
    assert (clip_levels[0][g_mask] == wb[1] * clip_scale).all(), "G photosites should clip at wb[1]"
    assert (clip_levels[0][b_mask] == wb[2] * clip_scale).all(), "B photosites should clip at wb[2]"

    # 7. clip_scale < 1 (highlight clip augmentation)
    clip_scale = 0.5
    clip_levels_scaled = wb[cfa.long()].unsqueeze(0) * clip_scale
    assert (clip_levels_scaled == clip_levels * 0.5).all(), "clip_scale should scale uniformly"

    # 8. Shape check: clip_ratio concatenates correctly
    masks = make_channel_masks(H, W, pattern)
    cfa_img = torch.zeros(1, H, W)
    clip_ratio = torch.zeros(1, H, W)
    input_tensor = torch.cat([cfa_img, masks, clip_ratio], dim=0)
    assert input_tensor.shape == (5, H, W), f"expected (5,H,W), got {input_tensor.shape}"

    # 9. Bayer pattern works too
    cfa_b = make_cfa_mask(16, 16, CFA_REGISTRY["bayer"])
    cl = wb[cfa_b.long()].unsqueeze(0)
    assert cl.shape == (1, 16, 16), "Bayer clip_levels shape wrong"
    # Bayer has 2x2 period — verify clip_levels tile correctly
    assert (cl[0, 0, 0] == cl[0, 2, 0]).item(), "Bayer clip_levels should repeat every 2 rows"
    assert (cl[0, 0, 0] == cl[0, 0, 2]).item(), "Bayer clip_levels should repeat every 2 cols"

    print("All clip_ratio tests passed!")


def test_live_dataset():
    """Test clip_ratio properties on actual dataset items."""
    from dataset import LinearDataset
    data_dir = "/Volumes/4T/RAISE/dataset"

    configs = [
        ("no augmentation",      dict(augment=False, apply_wb=False,
                                       noise_sigma=(0.0, 0.0), shot_noise=(0.0, 0.0))),
        ("wb + highlight aug",    dict(augment=False, apply_wb=True,
                                       highlight_aug_prob=1.0, highlight_aug_ev=2.0,
                                       noise_sigma=(0.0, 0.0), shot_noise=(0.0, 0.0))),
        ("wb + hl aug + geom",    dict(augment=True, apply_wb=True,
                                       highlight_aug_prob=1.0, highlight_aug_ev=2.0,
                                       noise_sigma=(0.0, 0.0), shot_noise=(0.0, 0.0))),
        ("noise only",           dict(augment=True, apply_wb=False,
                                       noise_sigma=(0.001, 0.01), shot_noise=(0.0, 0.005))),
    ]

    for name, kwargs in configs:
        print(f"\n--- {name} ---")
        ds = LinearDataset(data_dir=data_dir, patch_size=48, patches_per_image=1,
                           max_images=3, cfa_type="xtrans", **kwargs)

        for i in range(min(3, len(ds))):
            inp, ref, clip_ch = ds[i]

            # Shape checks
            assert inp.shape == (5, 48, 48), f"input shape {inp.shape}"
            assert ref.shape == (3, 48, 48), f"ref shape {ref.shape}"
            assert clip_ch.shape == (3,), f"clip_ch shape {clip_ch.shape}"

            cfa_img = inp[0:1]       # (1, H, W)
            masks = inp[1:4]         # (3, H, W)
            clip_ratio = inp[4:5]    # (1, H, W)

            # clip_ratio must be in [0, 1]
            assert clip_ratio.min() >= 0.0, f"clip_ratio min={clip_ratio.min()}"
            assert clip_ratio.max() <= 1.0, f"clip_ratio max={clip_ratio.max()}"

            # clip_ch must be positive
            assert (clip_ch > 0).all(), f"clip_ch has non-positive: {clip_ch}"

            # Masks should be binary and sum to 1 per pixel
            mask_sum = masks.sum(dim=0)
            assert (mask_sum == 1.0).all(), "masks should sum to 1 per pixel"

            # Reconstruct clip_levels from clip_ch and CFA mask
            cfa_mask = masks.argmax(dim=0)  # (H, W) with values 0,1,2
            clip_levels_recon = clip_ch[cfa_mask].unsqueeze(0)  # (1, H, W)

            # Re-derive clip_ratio from cfa_img and clip_levels
            raw_ratio_recon = (cfa_img / (clip_levels_recon + 1e-8)).clamp(0, 1)
            clip_ratio_recon = ((raw_ratio_recon - 0.5) * 2.0).clamp(0, 1)

            # If no noise was added, clip_ratio should match exactly.
            # With noise, cfa_img changed after clip_ratio was computed, so
            # we can only check the no-noise cases for exact match.
            if "noise" not in name:
                err = (clip_ratio - clip_ratio_recon).abs().max()
                assert err < 1e-4, f"[{name}] clip_ratio mismatch: max_err={err:.6f}"
                print(f"  sample {i}: clip_ratio exact match (max_err={err:.6f}), "
                      f"range=[{clip_ratio.min():.3f}, {clip_ratio.max():.3f}], "
                      f"clip_ch={clip_ch.tolist()}")
            else:
                # With noise, just verify ranges are sane
                print(f"  sample {i}: clip_ratio range=[{clip_ratio.min():.3f}, {clip_ratio.max():.3f}], "
                      f"cfa_img range=[{cfa_img.min():.4f}, {cfa_img.max():.4f}], "
                      f"clip_ch={clip_ch.tolist()}")

    print("\nAll live dataset tests passed!")


if __name__ == "__main__":
    test_clip_ratio()
    test_live_dataset()
