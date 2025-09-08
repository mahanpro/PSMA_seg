import torch

_FIELD_LIST = [
    "norm1.weight",
    "norm1.bias",
    "conv1.conv.weight",
    "norm2.weight",
    "norm2.bias",
    "conv2.conv.weight",
    # optional third unit inside RU if present would go here
]


def load_init_ckpt(model: torch.nn.Module, ckpt_path: str, verbose=True):
    """
    Loads external weights into `model` (supports WrapPlainNet, SegResNetText backbones, etc.).
    - Accepts raw state_dict or checkpoint dict with 'state_dict'.
    - Strips 'module.' prefixes.
    - If model has an attribute '.net' (your wrapper), loads into that.
    - Partially loads by (name, shape) match; prints a summary.
    """
    if not ckpt_path:
        return

    sd = torch.load(ckpt_path, map_location="cpu")
    if (
        isinstance(sd, dict)
        and "state_dict" in sd
        and all(isinstance(k, str) for k in sd["state_dict"])
    ):
        sd = sd["state_dict"]

    # strip common prefixes
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    # choose target module (unwrap wrapper if present)
    target = getattr(model, "net", model)

    msd = target.state_dict()
    compat = {k: v for k, v in sd.items() if (k in msd and msd[k].shape == v.shape)}
    msd.update(compat)
    missing = sorted(set(msd.keys()) - set(compat.keys()))
    unexpected = sorted(set(sd.keys()) - set(compat.keys()))
    target.load_state_dict(msd, strict=False)

    if verbose:
        print(
            f"[init-load] loaded {len(compat)} tensors into {target.__class__.__name__}"
        )
        if unexpected:
            print(
                f"[init-load] {len(unexpected)} unexpected (ignored), e.g. {unexpected[:6]}"
            )
        if missing:
            print(f"[init-load] {len(missing)} missing (kept init), e.g. {missing[:6]}")


def load_trained_ckpt(
    model: torch.nn.Module,
    ckpt_path: str,
    strict: bool = True,
    verbose: bool = True,
):
    """
    Load a *trained* checkpoint saved by train_ddp.save_ckpt(...) into `model`.
    - Accepts a checkpoint dict with 'model' (preferred) or 'state_dict'.
    - Strips 'module.' prefixes (DDP).
    - Does NOT unwrap '.net' (use a wrapper if your keys include 'net.').
    """
    if not ckpt_path:
        raise ValueError("ckpt_path is empty.")

    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict):
        if "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        elif "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]

    # strip DDP prefix
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    msg = model.load_state_dict(sd, strict=strict)

    if verbose:
        mk = getattr(msg, "missing_keys", [])
        uk = getattr(msg, "unexpected_keys", [])
        print(f"[load-trained] missing={len(mk)} unexpected={len(uk)}")
        if mk[:5]:
            print("  missing (first 5):", mk[:5])
        if uk[:5]:
            print("  unexpected (first 5):", uk[:5])
    return msg


def port_monai_to_segresnet_text(
    model_text: torch.nn.Module, monai_state: dict, verbose=True
):
    """
    Partially initialize SegResNetText encoder from a MONAI SegResNet checkpoint.
    Assumes same widths: init_filters=16, blocks_down=(1,2,2,4), etc.
    """
    td = model_text.state_dict()
    new_sd = {k: v.clone() for k, v in td.items()}
    loaded = 0

    # 0) stem
    if "convInit.conv.weight" in monai_state and "stem.conv1.conv.weight" in new_sd:
        if (
            new_sd["stem.conv1.conv.weight"].shape
            == monai_state["convInit.conv.weight"].shape
        ):
            new_sd["stem.conv1.conv.weight"] = monai_state["convInit.conv.weight"]
            loaded += 1

    # helper to copy a RU block by field list
    def _copy_block(src_prefix, dst_prefix):
        nonlocal loaded
        ok = False
        for f in _FIELD_LIST:
            ks, kd = f"{src_prefix}{f}", f"{dst_prefix}{f}"
            if (
                ks in monai_state
                and kd in new_sd
                and new_sd[kd].shape == monai_state[ks].shape
            ):
                new_sd[kd] = monai_state[ks]
                loaded += 1
                ok = True
        return ok

    # 1) stage 0 → enc1.(0..)
    b = 1
    t = 0
    while True:
        src_pref = f"down_layers.0.{b}."
        if not any(k.startswith(src_pref) for k in monai_state.keys()):
            break
        _copy_block(src_pref, f"enc1.{t}.")
        b += 1
        t += 1

    # 2) stages 1..3 → enc2..enc4
    for s in (1, 2, 3):
        enc_name = f"enc{s+1}"

        # stride-2 downsample conv → first RU conv1
        ks, kd = f"down_layers.{s}.0.conv.weight", f"{enc_name}.0.conv1.conv.weight"
        if (
            ks in monai_state
            and kd in new_sd
            and new_sd[kd].shape == monai_state[ks].shape
        ):
            new_sd[kd] = monai_state[ks]
            loaded += 1

        # remaining RUs
        b = 1
        t = 0
        while True:
            src_pref = f"down_layers.{s}.{b}."
            if not any(k.startswith(src_pref) for k in monai_state.keys()):
                break
            _copy_block(src_pref, f"{enc_name}.{t}.")
            b += 1
            t += 1

    model_text.load_state_dict(new_sd, strict=False)
    if verbose:
        print(
            f"[port] SegResNet (MONAI) → SegResNetText: loaded ~{loaded} encoder tensors (decoder left as init)."
        )
