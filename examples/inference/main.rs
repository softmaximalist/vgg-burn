use burn::backend::NdArray;
use burn::tensor::backend::Backend;
use burn::{
    Tensor,
    tensor::{Element, TensorData, activation::softmax},
};
use vgg_burn::{Vgg, model::imagenet};

const HEIGHT: usize = 224;
const WIDTH: usize = 224;

fn to_tensor<B: Backend, T: Element>(
    data: Vec<T>,
    shape: [usize; 3],
    device: &B::Device,
) -> Tensor<B, 3> {
    let img_tensor_data = TensorData::new(data, shape).convert::<B::FloatElem>();
    // [H, W, C] -> [C, H, W]
    // Divide by 255 to normalize pixel values between [0, 1]
    Tensor::from_data(img_tensor_data, device).permute([2, 0, 1]) / 255_f32
}

pub fn main() {
    // parse command line arguments
    let img_path = std::env::args().nth(1).expect("No image path provided");

    // Create VGG model
    let device = Default::default();
    let vgg: Vgg<NdArray> = Vgg::vgg16(false, true, &device);

    // Load image
    let img = image::open(&img_path)
        .map_err(|err| format!("Failed to load image {img_path}.\nError: {err}"))
        .unwrap();

    // Resize image to 224 by 224
    let resized_img = img.resize_exact(
        WIDTH as u32,
        HEIGHT as u32,
        image::imageops::FilterType::Triangle,
    );

    // Create tensor from image data
    let img_rgb = resized_img.into_rgb8();
    let img_tensor = to_tensor(img_rgb.into_raw(), [HEIGHT, WIDTH, 3], &device).unsqueeze::<4>();

    // Normalize the image using the ImageNet normalizer
    let input = imagenet::Normalizer::new(&device).normalize(img_tensor);
    
    // println!("input: {}", input);

    // Forward pass
    let out = vgg.forward(input.clone());
    let probs = softmax(out, 1);

    // Output class index with score
    let (score, index) = probs.max_dim_with_indices(1);
    let score = score.into_scalar();
    let index = index.into_scalar() as usize;
    println!(
        "Predicted: {}\nCategory Id: {}\nScore: {:.4}",
        imagenet::CLASSES[index],
        index,
        score
    )
}
