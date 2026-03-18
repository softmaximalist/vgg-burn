use burn::module::Module;
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::Tensor;
use super::conv_block::*;
use super::fc_block::*;
#[cfg(feature = "pretrained")]
use super::weights::load_pretrained_weights;

#[cfg(not(feature = "pretrained"))]
fn load_pretrained_weights<B: Backend>(_vgg: Vgg<B>, _version: VggVersion) -> Vgg<B> {
    panic!("Cannot load pretrained weights. Please enable the 'pretrained' feature in your Cargo.toml.");
}

pub enum VggVersion {
    Vgg11,
    Vgg11Bn,
    Vgg13,
    Vgg13Bn,
    Vgg16,
    Vgg16Bn,
    Vgg19,
    Vgg19Bn
}

#[derive(Module, Debug)]
pub struct Vgg<B: Backend> {
    conv_block1: ConvBlock<B>,
    conv_block2: ConvBlock<B>,
    conv_block3: ConvBlock<B>,
    conv_block4: ConvBlock<B>,
    conv_block5: ConvBlock<B>,
    fc_block: FcBlock<B>
}

impl<B: Backend> Vgg<B> {
    fn construct_vgg(layer_nums: [usize; 5], batch_normalize: bool, device: &B::Device) -> Self {
        Self {
            conv_block1: ConvBlock::new(64, layer_nums[0], batch_normalize, device),
            conv_block2: ConvBlock::new(128, layer_nums[1], batch_normalize, device),
            conv_block3: ConvBlock::new(256, layer_nums[2], batch_normalize, device),
            conv_block4: ConvBlock::new(512, layer_nums[3], batch_normalize, device),
            conv_block5: ConvBlock::new(512, layer_nums[4], batch_normalize, device),
            fc_block: FcBlock::new(device)
        }
    }
    
    pub fn vgg11(batch_normalize: bool, pretrained: bool, device: &B::Device) -> Self {
        let vgg = Self::construct_vgg([1_usize, 1, 2, 2, 2], batch_normalize, device);
        if pretrained {
            if batch_normalize {
                return load_pretrained_weights(vgg, VggVersion::Vgg11Bn)
            } else {
                return load_pretrained_weights(vgg, VggVersion::Vgg11)
            }
        }
        
        vgg
    }
    
    pub fn vgg13(batch_normalize: bool, pretrained: bool, device: &B::Device) -> Self {
        let vgg = Self::construct_vgg([2_usize, 2, 2, 2, 2], batch_normalize, device);
        if pretrained {
            if batch_normalize {
                return load_pretrained_weights(vgg, VggVersion::Vgg13Bn)
            } else {
                return load_pretrained_weights(vgg, VggVersion::Vgg13)
            }
        }
        
        vgg
    }
    
    pub fn vgg16(batch_normalize: bool, pretrained: bool, device: &B::Device) -> Self {
        let vgg = Self::construct_vgg([2_usize, 2, 3, 3, 3], batch_normalize, device);
        if pretrained {
            if batch_normalize {
                return load_pretrained_weights(vgg, VggVersion::Vgg16Bn)
            } else {
                return load_pretrained_weights(vgg, VggVersion::Vgg16)
            }
        }
        
        vgg
    }
    
    pub fn vgg19(batch_normalize: bool, pretrained: bool, device: &B::Device) -> Self {
        let vgg = Self::construct_vgg([2_usize, 2, 4, 4, 4], batch_normalize, device);
        if pretrained {
            if batch_normalize {
                return load_pretrained_weights(vgg, VggVersion::Vgg19Bn)
            } else {
                return load_pretrained_weights(vgg, VggVersion::Vgg19)
            }
        }
        
        vgg
    }
    
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let [n, c, _, _] = input.dims();
        let reshaped_input = input.reshape([n, c, 224, 224]);
        
        let conv1_out = self.conv_block1.forward(reshaped_input);
        let conv2_out = self.conv_block2.forward(conv1_out);
        let conv3_out = self.conv_block3.forward(conv2_out);
        let conv4_out = self.conv_block4.forward(conv3_out);
        let conv5_out = self.conv_block5.forward(conv4_out);
        let conv_out = conv5_out.flatten(1, 3);
        
        let fc_out = self.fc_block.forward(conv_out);
        softmax(fc_out, 1)
    }
}