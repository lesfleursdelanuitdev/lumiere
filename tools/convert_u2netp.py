#!/usr/bin/env python3
"""Convert U²-Net-p (ONNX) into the compact JSON graph that index.html runs with TensorFlow.js.

The app does not ship an ONNX runtime. Instead it runs the network with TensorFlow.js ops on the
GPU (WebGL), so the browser never has to compile WebAssembly. This script:

  1. keeps only the operators that carry image data (Conv, Relu, MaxPool, Resize, Concat, Add,
     Sigmoid) and drops the shape bookkeeping around them;
  2. records each Resize's output size, found by running the model once with ONNX Runtime;
  3. stores the weights as float16 (base64) in NHWC filter order, about 3 MB in total.

When the model was converted for this repository, the TensorFlow.js output matched ONNX Runtime
to a mean absolute difference of 0.0003 (the gap comes from storing weights as float16).

Usage:
    pip install onnx onnxruntime numpy
    curl -L -o u2netp.onnx https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx
    python3 tools/convert_u2netp.py u2netp.onnx models/u2netp.json

U²-Net: Xuebin Qin et al., "U²-Net: Going Deeper with Nested U-Structure for Salient Object
Detection", Pattern Recognition 2020. Apache-2.0. See THIRD_PARTY_NOTICES.md.
"""
import argparse
import base64
import json

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

KEEP = {"Conv", "Relu", "MaxPool", "Resize", "Concat", "Add", "Sigmoid"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx_path")
    ap.add_argument("out_json")
    ap.add_argument("--input", default="input.1", help="name of the image input")
    ap.add_argument("--output", default="1959", help="name of the fused saliency output (d0)")
    args = ap.parse_args()

    model = onnx.load(args.onnx_path)
    graph = model.graph
    inits = {t.name: numpy_helper.to_array(t) for t in graph.initializer}

    # Operators fed (directly or indirectly) by the image.
    feat, kept = {args.input}, []
    for n in graph.node:
        if n.op_type in KEEP and n.input and n.input[0] in feat:
            if n.op_type == "Concat" and not all(i in feat for i in n.input):
                continue
            kept.append(n)
            feat.update(n.output)

    # Run once with every kept tensor exposed, to learn Resize output sizes.
    probe = onnx.load(args.onnx_path)
    have = {o.name for o in probe.graph.output}
    for n in kept:
        for o in n.output:
            if o not in have:
                probe.graph.output.append(helper.make_tensor_value_info(o, onnx.TensorProto.FLOAT, None))
    sess = ort.InferenceSession(probe.SerializeToString(), providers=["CPUExecutionProvider"])
    x = np.random.RandomState(0).rand(1, 3, 320, 320).astype(np.float32)
    names = [o.name for o in sess.get_outputs()]
    values = dict(zip(names, sess.run(None, {args.input: x})))

    ids = {args.input: 0}

    def tid(name):
        if name not in ids:
            ids[name] = len(ids)
        return ids[name]

    blobs, offset, ops = [], 0, []

    def add_blob(arr):
        nonlocal offset
        a = arr.astype(np.float16).ravel()
        blobs.append(a)
        start, offset = offset, offset + a.size
        return [start, int(a.size)]

    for n in kept:
        a = {at.name: helper.get_attribute_value(at) for at in n.attribute}
        op = {"op": n.op_type, "out": tid(n.output[0])}
        if n.op_type == "Conv":
            w, b = inits[n.input[1]], inits[n.input[2]]
            k, d = w.shape[2], a["dilations"][0]
            assert a.get("group", 1) == 1 and a["strides"] == [1, 1]
            assert a["pads"] == [d * (k - 1) // 2] * 4, "only 'same' padding is supported"
            op.update(inp=tid(n.input[0]), w=add_blob(w.transpose(2, 3, 1, 0)), wshape=[k, k, int(w.shape[1]), int(w.shape[0])], b=add_blob(b), dil=d)
        elif n.op_type == "MaxPool":
            assert a["kernel_shape"] == [2, 2] and a["strides"] == [2, 2] and a["ceil_mode"] == 1
            op.update(inp=tid(n.input[0]))
        elif n.op_type == "Resize":
            assert a["mode"] == b"linear" and a["coordinate_transformation_mode"] == b"pytorch_half_pixel"
            shape = values[n.output[0]].shape
            op.update(inp=tid(n.input[0]), size=[int(shape[2]), int(shape[3])])
        elif n.op_type in ("Concat", "Add"):
            if n.op_type == "Concat":
                assert a["axis"] == 1
            op.update(inps=[tid(i) for i in n.input])
        else:
            op.update(inp=tid(n.input[0]))
        ops.append(op)

    weights = np.concatenate(blobs)
    out = {
        "name": "u2netp",
        "license": "Apache-2.0 (U-2-Net, Qin et al. 2020)",
        "input": [320, 320],
        "output": ids[args.output],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "ops": ops,
        "weights": base64.b64encode(weights.tobytes()).decode(),
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"{len(ops)} ops, {weights.size} weights, {len(json.dumps(out)) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
