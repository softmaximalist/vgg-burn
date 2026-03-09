use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::Tensor;

pub struct FcBlock<B: Backend> {
    layer1: Linear<B>,
    layer2: Linear<B>,
    layer3: Linear<B>,
}

impl<B: Backend> FcBlock<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            layer1: LinearConfig::new(25088, 4096).init(device),
            layer2: LinearConfig::new(4096, 4096).init(device),
            layer3: LinearConfig::new(4096, 1000).init(device),
        }
    }
    
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let layer1_out = self.layer1.forward(input);
        let layer2_out = self.layer2.forward(layer1_out);
        self.layer3.forward(layer2_out)
    }
}