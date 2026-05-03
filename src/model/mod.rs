mod conv_block;
mod fc_block;
pub mod imagenet;
pub mod vgg;
#[cfg(feature = "pretrained")]
mod weights;

pub use vgg::*;
