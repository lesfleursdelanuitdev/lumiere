# Third-party notices

## U²-Net (model weights in `models/u2netp.json`)

- Xuebin Qin, Zichen Zhang, Chenyang Huang, Masood Dehghan, Osmar R. Zaiane and Martin Jagersand,
  "U²-Net: Going Deeper with Nested U-Structure for Salient Object Detection", *Pattern Recognition* 106, 2020.
- Source: <https://github.com/xuebinqin/U-2-Net>, Apache License 2.0.
- The ONNX export used for conversion is the `u2netp.onnx` file distributed by rembg
  (<https://github.com/danielgatis/rembg>). `tools/convert_u2netp.py` converts it to the
  JSON graph in this repository; weights are stored as float16 and are otherwise unchanged.

## SAM 2 (model weights in `models/sam2-tiny/`, port in `sam2.js`)

- Nikhila Ravi et al. (Meta FAIR), "SAM 2: Segment Anything in Images and Videos", 2024.
- Source and weights: <https://github.com/facebookresearch/sam2>, Apache License 2.0
  (Copyright Meta Platforms, Inc. and affiliates).
- The checkpoint is SAM 2.1 Hiera Tiny. It was obtained as `sam2.1_t.pt` from
  <https://github.com/ultralytics/assets/releases> (a re-hosted copy of Meta's weights; it loads
  strictly into Meta's model definition) and converted with `tools/convert_sam2.py`: only the
  image encoder, prompt encoder and mask decoder are kept, two pairs of 1×1 convolutions are
  multiplied together, and weights are stored as float16. `sam2.js` is a TensorFlow.js
  re-implementation of Meta's PyTorch code for these parts.

## TensorFlow.js

- Loaded at run time from jsDelivr (`@tensorflow/tfjs@4.22.0`), not included in this repository.
- <https://github.com/tensorflow/tfjs>, Apache License 2.0.

## Fonts

Loaded at run time from Google Fonts, not included in this repository. All three are licensed
under the SIL Open Font License 1.1:

- Manrope
- IBM Plex Sans
- IBM Plex Mono

## Algorithms

- Guided filter for edge-aware mask refinement: Kaiming He, Jian Sun and Xiaoou Tang,
  "Guided Image Filtering", ECCV 2010 / IEEE TPAMI 2013. Implemented from the paper.
