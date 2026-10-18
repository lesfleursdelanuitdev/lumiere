# Third-party notices

## U²-Net (model weights in `models/u2netp.json`)

- Xuebin Qin, Zichen Zhang, Chenyang Huang, Masood Dehghan, Osmar R. Zaiane and Martin Jagersand,
  "U²-Net: Going Deeper with Nested U-Structure for Salient Object Detection", *Pattern Recognition* 106, 2020.
- Source: <https://github.com/xuebinqin/U-2-Net>, Apache License 2.0.
- The ONNX export used for conversion is the `u2netp.onnx` file distributed by rembg
  (<https://github.com/danielgatis/rembg>). `tools/convert_u2netp.py` converts it to the
  JSON graph in this repository; weights are stored as float16 and are otherwise unchanged.

## TensorFlow.js

- Loaded at run time from jsDelivr (`@tensorflow/tfjs@4.22.0`), not included in this repository.
- <https://github.com/tensorflow/tfjs>, Apache License 2.0.

## Fonts

Loaded at run time from Google Fonts, not included in this repository. All three are licensed
under the SIL Open Font License 1.1:

- Cormorant Garamond
- Figtree
- JetBrains Mono

## Algorithms

- Guided filter for edge-aware mask refinement: Kaiming He, Jian Sun and Xiaoou Tang,
  "Guided Image Filtering", ECCV 2010 / IEEE TPAMI 2013. Implemented from the paper.
