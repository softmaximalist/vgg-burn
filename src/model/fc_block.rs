use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig, Relu};
use burn::tensor::backend::Backend;
use burn::Tensor;

#[derive(Module, Debug)]
pub struct FcBlock<B: Backend> {
    layer1: Linear<B>,
    layer2: Linear<B>,
    layer3: Linear<B>,
    relu: Relu,
    dropout: Dropout,
}

impl<B: Backend> FcBlock<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            layer1: LinearConfig::new(25088, 4096).init(device),
            layer2: LinearConfig::new(4096, 4096).init(device),
            layer3: LinearConfig::new(4096, 1000).init(device),
            relu: Relu,
            dropout: DropoutConfig::new(0.5).init()
        }
    }
    
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let layer1_out = self.layer1.forward(input);
        let relu1_out = self.relu.forward(layer1_out);
        let dropout1_out = self.dropout.forward(relu1_out);
        
        let layer2_out = self.layer2.forward(dropout1_out);
        let relu2_out = self.relu.forward(layer2_out);
        let dropout2_out = self.dropout.forward(relu2_out);
        
        self.layer3.forward(dropout2_out)
    }
}