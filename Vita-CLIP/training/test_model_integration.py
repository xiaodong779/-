#!/usr/bin/env python
"""End-to-end minimal integration test for Vita-CLIP (Stage 10).

Covers:
  1. model construction with new modules (frequency adapter + two text branches)
  2. full forward [B,3,T,224,224] -> logits
  3. backward, gradients on new modules non-zero, frozen params no grad
  4. frequency adapter only runs at configured layers (forward-hook counter)
  5. original visual prompts not executed (when disabled)
  6. fixed branch and learnable branch both run
  7. fallback: all new modules off -> baseline forward still works
  8. restore: use_original_visual_prompts=True still builds
  9. checkpoint strict=False loading (missing/unexpected reported)
  10. AMP forward/backward (if CUDA)
"""
import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')

from VitaCLIP_model import VitaCLIP


CLASSES_PATH = '../classes/k400_classes.txt'
BACKBONE = os.environ.get('VITACLIP_BACKBONE', '')


def check_nan_inf(tag, t):
    n_nan = int(torch.isnan(t).sum())
    n_inf = int(torch.isinf(t).sum())
    assert n_nan == 0 and n_inf == 0, f'{tag} has NaN/Inf'
    print(f'  [{tag}] nan={n_nan} inf={n_inf}')


def build_model(**overrides):
    kw = dict(
        input_size=(224, 224),
        num_frames=8,
        feature_dim=768,
        patch_size=(16, 16),
        num_heads=12,
        num_layers=12,
        mlp_factor=4.0,
        embed_dim=512,
        # original visual prompts OFF by default
        use_original_visual_prompts=False,
        use_summary_token=False,
        use_local_prompts=False,
        use_global_prompts=False,
        num_global_prompts=8,
        # frequency adapter
        use_frequency_adapter=True,
        frequency_layers=(3, 7, 11),
        share_frequency_adapter=False,
        frequency_bottleneck=192,
        # text
        use_text_prompt_learning=True,
        text_context_length=77,
        text_vocab_size=49408,
        text_transformer_width=512,
        text_transformer_heads=8,
        text_transformer_layers=12,
        text_num_prompts=8,
        text_prompt_pos='end',
        text_prompt_init='',
        text_prompt_CSC=False,
        text_prompt_classes_path=CLASSES_PATH,
        use_fixed_text_branch=True,
        use_learnable_text_branch=True,
        use_text_mstff=True,
        fixed_text_template='a video of {}',
        zeroshot_evaluation=False,
        zeroshot_text_features_path='',
    )
    kw.update(overrides)
    return VitaCLIP(**kw)


def count_frequency_adapter_calls(model):
    """Count calls of each frequency adapter via forward hooks."""
    counts = {}

    def make_hook(layer_idx):
        def hook(mod, inp, out):
            counts[layer_idx] = counts.get(layer_idx, 0) + 1
        return hook

    handles = []
    if model.visual.use_frequency_adapter:
        if model.visual.share_frequency_adapter:
            h = model.visual.frequency_adapter_shared.register_forward_hook(
                make_hook('shared'))
            handles.append(h)
        else:
            for idx, mod in model.visual.frequency_adapters.items():
                h = mod.register_forward_hook(make_hook(int(idx)))
                handles.append(h)
    return counts, handles


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device={device}')

    # ------------------------------------------------------------
    # 1. Full model with new modules
    # ------------------------------------------------------------
    print('\n=== 1. build model (frequency + two text branches + MSTFF) ===')
    model = build_model()
    if BACKBONE:
        ckpt = torch.load(BACKBONE, map_location='cpu')
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print(f'  backbone missing={len(missing)} unexpected={len(unexpected)}')
        for k in missing:
            if 'frequency_adapter' in k or 'text_mstff' in k or 'text_fusion' in k \
               or 'fixed_template' in k or 'prompt_learner' in k:
                continue
            print(f'    - {k}')
    model = model.to(device)

    # trainable / frozen audit
    trainable, frozen = [], []
    for name, p in model.named_parameters():
        (trainable if p.requires_grad else frozen).append(name)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f'  trainable params: {n_trainable} ({len(trainable)} tensors)')
    print(f'  frozen params: {n_frozen} ({len(frozen)} tensors)')
    for key in ['visual.blocks', 'visual.patch_embed', 'visual.pos_embed',
                'visual.ln_pre', 'visual.ln_post', 'visual.proj',
                'textual.']:
        assert not any(k.startswith(key) and p.requires_grad
                       for k, p in model.named_parameters()), f'{key} not frozen!'
    print('  CLIP encoder params frozen: OK')

    # ------------------------------------------------------------
    # 2. forward + shape trace
    # ------------------------------------------------------------
    print('\n=== 2. forward [B,3,T,224,224] ===')
    B, T = 2, 8
    x = torch.randn(B, 3, T, 224, 224, device=device)
    logits = model(x)
    print(f'  logits shape={tuple(logits.shape)} (expect [B, n_cls])')
    check_nan_inf('logits', logits)
    assert logits.shape[0] == B and logits.shape[1] > 1

    # ------------------------------------------------------------
    # 3. backward + gradient audit
    # ------------------------------------------------------------
    print('\n=== 3. backward + gradients ===')
    loss = logits.softmax(dim=-1).mean()
    loss.backward()
    grad_ok = {
        'prompt_learner.ctx': False,
        'text_mstff.eot_proj.weight': False,
        'text_mstff.fusion_logits': False,
        'text_fusion_logits': False,
    }
    for name, p in model.named_parameters():
        if name in grad_ok:
            grad_ok[name] = p.grad is not None and p.grad.abs().max() > 0
    for name, mod in model.visual.frequency_adapters.items():
        p = mod.down_projection.weight
        key = f'frequency_adapter.layer{name}.down.weight'
        grad_ok[key] = p.grad is not None and p.grad.abs().max() > 0
        print(f'  grad[{key}] maxabs='
              f'{p.grad.abs().max().item() if p.grad is not None else None}')
    for k, v in grad_ok.items():
        assert v, f'no gradient on {k}'
        print(f'  grad[{k}] = non-zero: {v}')
    # frozen params must have no grad
    bad_frozen = [n for n, p in model.named_parameters()
                  if not p.requires_grad and p.grad is not None]
    assert not bad_frozen, f'frozen params got grads: {bad_frozen}'
    print('  frozen params have no grad: OK')

    # ------------------------------------------------------------
    # 4. frequency adapter call counters
    # ------------------------------------------------------------
    print('\n=== 4. frequency adapter runs only at [3,7,11] ===')
    counts, handles = count_frequency_adapter_calls(model)
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    print(f'  frequency adapter calls: {dict(sorted(counts.items()))}')
    assert counts.get(3) == 1 and counts.get(7) == 1 and counts.get(11) == 1
    assert all(k in (3, 7, 11) for k in counts), 'unexpected layer called!'

    # ------------------------------------------------------------
    # 5. original prompts disabled -> no global/local/summary params used
    # ------------------------------------------------------------
    print('\n=== 5. original visual prompts disabled ===')
    assert model.visual.use_global_prompts is False
    assert model.visual.use_summary_token is False
    assert model.visual.use_local_prompts is False
    assert not hasattr(model.visual, 'global_prompts')
    print('  global/local/summary prompts: not built / not executed')

    # ------------------------------------------------------------
    # 6. two text branches both executed
    # ------------------------------------------------------------
    print('\n=== 6. two text branches ===')
    fixed_feats = model._encode_fixed_branch()
    learn_feats = model._encode_learnable_branch()
    print(f'  fixed branch: {tuple(fixed_feats.shape)} '
          f'(n_cls, embed_dim)')
    print(f'  learnable branch: {tuple(learn_feats.shape)}')
    assert fixed_feats.shape == learn_feats.shape == (logits.shape[1], 512)
    with torch.no_grad():
        prompts = model.prompt_learner()
    print(f'  prompt_learner ctx shape={tuple(prompts.shape)} '
          f'contains class+EOT tokens: n_ctx={prompts.shape[1]}')

    # ------------------------------------------------------------
    # 7. fallback: all new modules off (baseline)
    # ------------------------------------------------------------
    print('\n=== 7. fallback: new modules OFF ===')
    model_off = build_model(
        use_frequency_adapter=False,
        use_text_mstff=False,
        use_fixed_text_branch=False,
        use_learnable_text_branch=True,
        use_original_visual_prompts=False,
    ).to(device)
    with torch.no_grad():
        logits_off = model_off(x)
    print(f'  fallback logits={tuple(logits_off.shape)}')
    check_nan_inf('fallback_logits', logits_off)
    assert not model_off.visual.use_frequency_adapter
    assert not hasattr(model_off, 'text_mstff')

    # ------------------------------------------------------------
    # 8. restore: use_original_visual_prompts=True
    # ------------------------------------------------------------
    print('\n=== 8. restore original visual prompts ===')
    model_orig = build_model(
        use_original_visual_prompts=True,
        use_global_prompts=True,
        use_summary_token=True,
        use_local_prompts=True,
    ).to(device)
    with torch.no_grad():
        logits_orig = model_orig(x)
    print(f'  restored logits={tuple(logits_orig.shape)}')
    check_nan_inf('restored_logits', logits_orig)
    assert model_orig.visual.use_global_prompts is True

    # ------------------------------------------------------------
    # 9. checkpoint non-strict load round trip
    # ------------------------------------------------------------
    print('\n=== 9. checkpoint save/load (strict=False) ===')
    torch.save({'model': model.state_dict()}, '/tmp/vitaclip_test_ckpt.pth')
    ckpt = torch.load('/tmp/vitaclip_test_ckpt.pth', map_location='cpu')
    model2 = build_model().to(device)
    missing, unexpected = model2.load_state_dict(ckpt['model'], strict=False)
    print(f'  missing={missing} unexpected={unexpected}')
    assert not missing and not unexpected, 'same-arch checkpoint must restore exactly'
    # simulate loading an OLD checkpoint (without new modules)
    old_sd = {k: v for k, v in model2.state_dict().items()
              if 'frequency_adapter' not in k and 'text_mstff' not in k
              and 'text_fusion_logits' not in k
              and 'fixed_template_text_features' not in k}
    missing, unexpected = model2.load_state_dict(old_sd, strict=False)
    print(f'  old-ckpt missing={len(missing)} unexpected={len(unexpected)}')
    assert len(missing) > 0, 'expected missing keys for new modules'

    # ------------------------------------------------------------
    # 10. AMP (if CUDA)
    # ------------------------------------------------------------
    if device == 'cuda':
        print('\n=== 10. AMP forward/backward ===')
        model_amp = build_model().to(device)
        opt = torch.optim.AdamW(model_amp.parameters(), lr=1e-4)
        opt.zero_grad()
        with torch.cuda.amp.autocast():
            out = model_amp(x)
            loss_amp = out.softmax(dim=-1).mean()
        loss_amp.backward()
        opt.step()
        check_nan_inf('amp_logits', out)
        print('  AMP OK')
    else:
        print('\n=== 10. AMP skipped (no CUDA) ===')

    print('\n[PASS] model integration test OK')


if __name__ == '__main__':
    main()
