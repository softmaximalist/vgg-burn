mod conv_block;
mod fc_block;
#[cfg(feature = "pretrained")]
mod weights;
pub mod imagenet;
pub mod vgg;

pub use vgg::*;