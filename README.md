# VGG Burn

This repo contains the Burn implementation for the VGG models. Supported versions include:
- VGG11
- VGG11 (with batch norm)
- VGG13
- VGG13 (with batch norm)
- VGG16
- VGG16 (with batch norm)
- VGG19
- VGG19 (with batch norm)

Users can initialize the models with random weights or they can use versions with pretrained ImageNet weights loaded. 

### Usage

Add the following to your `Cargo.toml` file:
```text
[dependencies]
vgg-burn = { git = "https://github.com/softmaximalist/vgg-burn", default-features = false }

```
If you want to get the pretrained ImageNet weights, enable the `pretrained` feature flag:
```
[dependencies]
vgg-burn = { git = "https://github.com/softmaximalist/vgg-burn", features = ["pretrained"] }
```

### Example Usage

##### Inference
The inference example initializes a VGG16 model from the pretrained ImageNet weights with the NdArray backend and performs inference on the provided input image.

The example can be run using the following command:
```
cargo run --release --example inference samples/pineapple.jpg
```