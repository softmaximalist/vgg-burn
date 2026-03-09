use burn::Tensor;
use burn::nn::{BatchNorm, BatchNormConfig, PaddingConfig2d, Relu};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::tensor::backend::Backend;

pub struct ConvBlock<B: Backend> {
    pub conv_layers: Vec<Conv2d<B>>,
    pub relu: Relu,
    pub max_pool2d: MaxPool2d,
    pub batch_norm: Option<BatchNorm<B>>,
}

impl<B: Backend> ConvBlock<B> {
    pub fn new(channels: usize, layers_num: usize, batch_normalize: bool, device: &B::Device) -> Self {
        let mut conv_layers = Vec::with_capacity(layers_num);
        
        // Add first conv layer in the block
        let conv1_config;
        if channels == 64 {
            conv1_config = Self::conv_config(3, channels, device);
        } else {
            conv1_config = Self::conv_config(channels / 2, channels, device);
        }
        conv_layers.push(conv1_config);
        
        // Add the remaining conv layers in the block (if there are any)
        for _ in 0..layers_num-1 {
            let conv_config = Self::conv_config(channels, channels, device);
            conv_layers.push(conv_config);
        }
        
        // Initialize batch norm
        let batch_norm;
        if batch_normalize {
            batch_norm = Some(BatchNormConfig::new(channels).init(device));
        } else {
            batch_norm = None;
        }
        
        Self { 
            conv_layers,
            relu: Relu,
            max_pool2d: MaxPool2dConfig::new([2, 2]).init(),
            batch_norm
        }
    }
    
    fn conv_config(in_ch: usize, out_ch: usize, device: &B::Device) -> Conv2d<B> {
        Conv2dConfig::new([in_ch, out_ch], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Same)
            .init(device)
    }
    
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let [n, c, _, _] = input.dims();
        let reshaped_input = input.reshape([n, c, 224, 224]);
        
        let mut relu_out = reshaped_input;
        for conv_layer in &self.conv_layers {
            let mut conv_out = conv_layer.forward(relu_out);
            
            if let Some(bn) = &self.batch_norm {
                conv_out = bn.forward(conv_out);
            }
            
            relu_out = self.relu.forward(conv_out);
        }
        
        self.max_pool2d.forward(relu_out)
    }
}