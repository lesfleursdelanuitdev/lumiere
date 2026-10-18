#!/usr/bin/env python3
"""Convert SAM 2.1 Hiera Tiny into the compact format Lumière loads in the browser.

Keeps only what single-photo selection needs (image encoder, prompt encoder, mask decoder),
drops the video memory modules, folds two pairs of 1x1 convolutions together, lays every
weight out the way TensorFlow.js wants it, and writes:

    models/sam2-tiny/manifest.json        tensor names, shapes, types and offsets
    models/sam2-tiny/sam2-tiny-NN.bin     the weights, in chunks of at most 10 MB

Weights are stored as float16 (about 62 MB), which gives the same masks as float32. --int8
stores the image encoder's large matrices as int8 with one float16 scale per 32 weights (about
37 MB); on test photos that changed confident masks very little but uncertain ones (a box, or
several taps) by up to about 13%, so it is off by default.

With --reference IMAGE it also runs the PyTorch model on that image and writes reference
inputs and outputs (to --ref-out) for checking the JavaScript port; with --compare it reports
how much quantization changes the masks.

Needs: torch, numpy, pillow, and Meta's SAM 2 source (https://github.com/facebookresearch/sam2,
Apache-2.0) on --sam2-src. The checkpoint is SAM 2.1 Hiera Tiny (Apache-2.0), for example
sam2.1_hiera_tiny.pt from Meta or sam2.1_t.pt (the same weights) from ultralytics/assets.
"""
import argparse, json, os, sys
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("checkpoint")
p.add_argument("--sam2-src", required=True, help="folder that contains the sam2 package")
p.add_argument("--out", default="models/sam2-tiny")
p.add_argument("--int8", action="store_true", help="smaller (about 37 MB) but changes uncertain masks; see --compare")
p.add_argument("--chunk-mb", type=float, default=10)
p.add_argument("--reference", help="image to compute reference outputs for")
p.add_argument("--ref-out", default="sam2-reference")
p.add_argument("--compare", action="store_true", help="report mask agreement of the stored weights vs float32")
args = p.parse_args()

sys.path.insert(0, args.sam2_src)
import torch
from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
from sam2.modeling.position_encoding import PositionEmbeddingSine
from sam2.modeling.sam.prompt_encoder import PromptEncoder
from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.transformer import TwoWayTransformer

# ── the model, configured as sam2.1_hiera_t.yaml ──
def build():
    trunk = Hiera(embed_dim=96, num_heads=1, stages=[1, 2, 7, 2], global_att_blocks=[5, 7, 9], window_pos_embed_bkg_spatial_size=[7, 7])
    neck = FpnNeck(position_encoding=PositionEmbeddingSine(num_pos_feats=256, normalize=True, scale=None, temperature=10000),
                   d_model=256, backbone_channel_list=[768, 384, 192, 96], fpn_top_down_levels=[2, 3], fpn_interp_model="nearest")
    enc = ImageEncoder(trunk=trunk, neck=neck, scalp=1)
    pe = PromptEncoder(embed_dim=256, image_embedding_size=(64, 64), input_image_size=(1024, 1024), mask_in_chans=16)
    dec = MaskDecoder(num_multimask_outputs=3, transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
                      transformer_dim=256, iou_head_depth=3, iou_head_hidden_dim=256, use_high_res_features=True,
                      iou_prediction_use_sigmoid=True, pred_obj_scores=True, pred_obj_scores_mlp=True, use_multimask_token_for_obj_ptr=True)
    return enc, pe, dec

ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
if not isinstance(sd, dict):  # a pickled module
    sd = sd.state_dict()
enc, pe, dec = build()
def load(mod, prefix):
    sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    missing, unexpected = mod.load_state_dict(sub, strict=False)
    missing = [m for m in missing if not m.startswith("neck.position_encoding")]
    if missing or unexpected: raise SystemExit(f"{prefix}: missing {missing[:5]} unexpected {unexpected[:5]}")
load(enc, "image_encoder."); load(pe, "sam_prompt_encoder."); load(dec, "sam_mask_decoder.")
no_mem = sd["no_mem_embed"].reshape(256).float()
for m in (enc, pe, dec): m.eval()

# ── weights in TensorFlow.js layout ──
T = {}
f = lambda t: t.detach().float().cpu().numpy()
def lin(name, mod): T[name + ".w"] = f(mod.weight).T.copy(); T[name + ".b"] = f(mod.bias)
def ln(name, mod): T[name + ".g"] = f(mod.weight); T[name + ".b"] = f(mod.bias)
tr = enc.trunk
T["patch.w"] = f(tr.patch_embed.proj.weight).transpose(2, 3, 1, 0).copy()   # OIHW → HWIO
T["patch.b"] = f(tr.patch_embed.proj.bias)
T["pos"] = f(tr.pos_embed)[0]; T["pos_win"] = f(tr.pos_embed_window)[0]       # C×7×7, C×8×8
for i, b in enumerate(tr.blocks):
    k = f"blk{i}"
    ln(k + ".n1", b.norm1); ln(k + ".n2", b.norm2)
    lin(k + ".qkv", b.attn.qkv); lin(k + ".proj", b.attn.proj)
    lin(k + ".fc1", b.mlp.layers[0]); lin(k + ".fc2", b.mlp.layers[1])
    if b.dim != b.dim_out: lin(k + ".skip", b.proj)
conv1 = lambda c: (f(c.weight)[:, :, 0, 0].T.copy(), f(c.bias))                  # 1×1 conv as a matrix
def fuse(a, b):  # b(a(x)) for two 1×1 convs
    (wa, ba), (wb, bb) = conv1(a), conv1(b); return wa @ wb, ba @ wb + bb
nc = enc.neck.convs  # convs[0] reads the 768-channel level, convs[3] the 96-channel level
T["neck.s0.w"], T["neck.s0.b"] = fuse(nc[3].conv, dec.conv_s0)
T["neck.s1.w"], T["neck.s1.b"] = fuse(nc[2].conv, dec.conv_s1)
T["neck.l2.w"], T["neck.l2.b"] = conv1(nc[1].conv)
T["neck.l3.w"], T["neck.l3.b"] = conv1(nc[0].conv)
T["no_mem"] = f(no_mem)
T["pe.gauss"] = f(pe.pe_layer.positional_encoding_gaussian_matrix)
T["pe.points"] = np.concatenate([f(e.weight) for e in pe.point_embeddings])      # 4×256
T["pe.not_a_point"] = f(pe.not_a_point_embed.weight)[0]
T["pe.no_mask"] = f(pe.no_mask_embed.weight)[0]
tt = dec.transformer
def attn(name, a): lin(name + ".q", a.q_proj); lin(name + ".k", a.k_proj); lin(name + ".v", a.v_proj); lin(name + ".o", a.out_proj)
for i, L in enumerate(tt.layers):
    k = f"tw{i}"
    attn(k + ".self", L.self_attn); attn(k + ".t2i", L.cross_attn_token_to_image); attn(k + ".i2t", L.cross_attn_image_to_token)
    for j in range(1, 5): ln(f"{k}.n{j}", getattr(L, f"norm{j}"))
    lin(k + ".fc1", L.mlp.layers[0]); lin(k + ".fc2", L.mlp.layers[1])
attn("tw.final", tt.final_attn_token_to_image); ln("tw.nf", tt.norm_final_attn)
T["dec.tokens"] = np.concatenate([f(dec.obj_score_token.weight), f(dec.iou_token.weight), f(dec.mask_tokens.weight)])  # 6×256
def convT(name, c):  # 2×2 stride-2 transposed conv as a matrix: in → (i, j, out), then depth-to-space
    w = f(c.weight); cin, cout = w.shape[:2]
    T[name + ".w"] = w.transpose(0, 2, 3, 1).reshape(cin, 4 * cout).copy(); T[name + ".b"] = np.tile(f(c.bias), 4)
up = dec.output_upscaling
convT("up1", up[0]); ln("up.ln", up[1]); convT("up2", up[3])
for i, m in enumerate(dec.output_hypernetworks_mlps):
    for j, l in enumerate(m.layers): lin(f"hyper{i}.{j}", l)
for j, l in enumerate(dec.iou_prediction_head.layers): lin(f"iou.{j}", l)

# ── store ──
os.makedirs(args.out, exist_ok=True)
for fn in os.listdir(args.out):
    if fn.endswith(".bin"): os.remove(os.path.join(args.out, fn))
blob, entries, stored = bytearray(), [], {}
GROUP = 32
def put(arr):
    while len(blob) % 4: blob.append(0)
    off = len(blob); blob.extend(arr.tobytes()); return off
for name, a in T.items():
    a = np.ascontiguousarray(a, dtype=np.float32)
    e = {"name": name, "shape": list(a.shape)}
    if args.int8 and a.ndim == 2 and a.size >= 16384 and name.startswith(("blk", "neck")) and a.shape[0] % GROUP == 0:
        # image encoder matrices: symmetric int8 with one float16 scale per GROUP inputs of each output
        g = a.reshape(a.shape[0] // GROUP, GROUP, a.shape[1])
        s = (np.maximum(np.abs(g).max(axis=1), 1e-8) / 127.0).astype(np.float16).astype(np.float32)
        q = np.clip(np.round(g / s[:, None, :]), -127, 127).astype(np.int8)
        e.update(dtype="int8", group=GROUP, offset=put(q), scale=put(s.astype(np.float16)))
        stored[name] = (q.astype(np.float32) * s[:, None, :]).reshape(a.shape)
    else:
        h = a.astype(np.float16); e.update(dtype="float16", offset=put(h)); stored[name] = h.astype(np.float32)
    entries.append(e)
chunk = int(args.chunk_mb * 1024 * 1024); files = []
for i in range(0, len(blob), chunk):
    fn = f"sam2-tiny-{i // chunk:02d}.bin"; files.append(fn)
    with open(os.path.join(args.out, fn), "wb") as fh: fh.write(blob[i:i + chunk])
manifest = {"format": "lumiere-sam2", "version": 1,
            "model": "SAM 2.1 Hiera Tiny (Meta, Apache-2.0)", "image_size": 1024,
            "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
            "trunk": {"stages": [1, 2, 7, 2], "embed_dim": 96, "heads": 1, "global": [5, 7, 9], "windows": [8, 4, 14, 7]},
            "quantization": "image encoder matrices int8 with a float16 scale per 32 inputs; float16 otherwise" if args.int8 else "float16",
            "bytes": len(blob), "files": files, "tensors": entries}
with open(os.path.join(args.out, "manifest.json"), "w") as fh: json.dump(manifest, fh, separators=(",", ":"))
print(f"{len(entries)} tensors, {sum(a.size for a in T.values()) / 1e6:.1f} M parameters, {len(blob) / 1e6:.1f} MB in {len(files)} files → {args.out}")

# ── reference outputs and quantization check ──
if args.reference or args.compare:
    from PIL import Image
    img = Image.open(args.reference).convert("RGB")
    W0, H0 = img.size
    x = np.asarray(img.resize((1024, 1024), Image.BILINEAR), dtype=np.float32) / 255.0
    x = (x - np.array(manifest["mean"], np.float32)) / np.array(manifest["std"], np.float32)
    xt = torch.from_numpy(x).permute(2, 0, 1)[None]
    # Prompts in 1024×1024 model space: one tap, a tap plus an exclusion, a box, a box plus a tap.
    cases = [{"points": [[600, 430, 1]]}, {"points": [[600, 430, 1], [520, 260, 0]]},
             {"points": [[300, 200, 2], [800, 760, 3]]}, {"points": [[300, 200, 2], [800, 760, 3], [560, 500, 1]]}]

    @torch.no_grad()
    def run():
        out = enc(xt); fpn = out["backbone_fpn"]
        s0, s1 = dec.conv_s0(fpn[0]), dec.conv_s1(fpn[1]); emb = fpn[2] + no_mem.view(1, 256, 1, 1)
        res = []
        for c in cases:
            pts = torch.tensor([[q[:2] for q in c["points"]]], dtype=torch.float32); lab = torch.tensor([[q[2] for q in c["points"]]], dtype=torch.int32)
            sp, de = pe(points=(pts, lab), boxes=None, masks=None)
            m, iou, _, _ = dec.predict_masks(image_embeddings=emb, image_pe=pe.get_dense_pe(), sparse_prompt_embeddings=sp,
                                             dense_prompt_embeddings=de, repeat_image=False, high_res_features=[s0, s1])
            res.append((m[0].numpy(), iou[0].numpy()))
        return out, s0, s1, emb, res

    full = run()
    if args.reference:
        os.makedirs(args.ref_out, exist_ok=True)
        nhwc = lambda t: np.ascontiguousarray(t[0].permute(1, 2, 0).numpy(), dtype=np.float32)
        x.astype(np.float32).tofile(os.path.join(args.ref_out, "input.f32"))
        with torch.no_grad(): trunk_feats = enc.trunk(xt)
        for i, t in enumerate(trunk_feats): nhwc(t).tofile(os.path.join(args.ref_out, f"trunk{i}.f32"))
        nhwc(full[1]).tofile(os.path.join(args.ref_out, "s0.f32")); nhwc(full[2]).tofile(os.path.join(args.ref_out, "s1.f32")); nhwc(full[3]).tofile(os.path.join(args.ref_out, "embed.f32"))
        for i, (m, iou) in enumerate(full[4]):
            m.astype(np.float32).tofile(os.path.join(args.ref_out, f"case{i}.f32")); cases[i]["iou"] = iou.tolist()
        with open(os.path.join(args.ref_out, "cases.json"), "w") as fh: json.dump({"image": [W0, H0], "cases": cases}, fh)
        print("reference written to", args.ref_out)
    if args.compare:  # put the stored (rounded) weights back into the model and compare the masks
        def back(name, shape=None):
            a = stored[name]; return torch.from_numpy(a.reshape(shape) if shape else a)
        with torch.no_grad():
            for i, b in enumerate(tr.blocks):
                k = f"blk{i}"
                for nm, mod in [("qkv", b.attn.qkv), ("proj", b.attn.proj), ("fc1", b.mlp.layers[0]), ("fc2", b.mlp.layers[1])] + ([("skip", b.proj)] if b.dim != b.dim_out else []):
                    mod.weight.copy_(back(f"{k}.{nm}.w").T); mod.bias.copy_(back(f"{k}.{nm}.b"))
            for i, L in enumerate(tt.layers):
                k = f"tw{i}"
                for an, a in [("self", L.self_attn), ("t2i", L.cross_attn_token_to_image), ("i2t", L.cross_attn_image_to_token)]:
                    for pn in "qkvo":
                        mod = getattr(a, {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "out_proj"}[pn])
                        mod.weight.copy_(back(f"{k}.{an}.{pn}.w").T); mod.bias.copy_(back(f"{k}.{an}.{pn}.b"))
                L.mlp.layers[0].weight.copy_(back(f"{k}.fc1.w").T); L.mlp.layers[1].weight.copy_(back(f"{k}.fc2.w").T)
        q = run()
        for i, ((m0, i0), (m1, i1)) in enumerate(zip(full[4], q[4])):
            a, b = m0 > 0, m1 > 0
            print(f"case {i}: mask agreement (IoU) per output", [round(float((a[j] & b[j]).sum() / max(1, (a[j] | b[j]).sum())), 4) for j in range(4)],
                  "| predicted quality", np.round(i0, 3).tolist(), "→", np.round(i1, 3).tolist())
