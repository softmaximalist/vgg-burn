use burn::Tensor;
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, PaddingConfig2d, Relu};
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    pub conv_layers: Vec<Conv2d<B>>,
    pub relu: Relu,
    pub max_pool2d: MaxPool2d,
    pub bn_layers: Vec<BatchNorm<B>>,
}

impl<B: Backend> ConvBlock<B> {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        layers_num: usize,
        batch_normalize: bool,
        device: &B::Device,
    ) -> Self {
        let mut conv_layers = Vec::with_capacity(layers_num);
        let mut bn_layers = Vec::with_capacity(layers_num);

        // Add first conv layer in the block
        conv_layers.push(Self::conv_config(in_channels, out_channels, device));

        // Add first bn layer in the block if batch normalization is used
        if batch_normalize {
            bn_layers.push(BatchNormConfig::new(out_channels).init(device));
        }

        // Add the remaining conv layers and bn layers in the block (if there are any)
        for _ in 0..layers_num - 1 {
            let conv_config = Self::conv_config(out_channels, out_channels, device);
            conv_layers.push(conv_config);

            if batch_normalize {
                bn_layers.push(BatchNormConfig::new(out_channels).init(device));
            }
        }

        Self {
            conv_layers,
            relu: Relu,
            max_pool2d: MaxPool2dConfig::new([2, 2]).init(),
            bn_layers,
        }
    }

    fn conv_config(in_ch: usize, out_ch: usize, device: &B::Device) -> Conv2d<B> {
        Conv2dConfig::new([in_ch, out_ch], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Same)
            .init(device)
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut relu_out = input;
        if self.bn_layers.is_empty() {
            for conv_layer in &self.conv_layers {
                let conv_out = conv_layer.forward(relu_out);
                relu_out = self.relu.forward(conv_out);
            }
        } else {
            for (conv_layer, bn_layer) in self.conv_layers.iter().zip(&self.bn_layers) {
                let conv_out = conv_layer.forward(relu_out);
                let bn_out = bn_layer.forward(conv_out);
                relu_out = self.relu.forward(bn_out);
            }
        }

        self.max_pool2d.forward(relu_out)
    }
}
