use std::{fs::{self, File, rename}, io::Write, path::PathBuf};
use burn_std::network::downloader::download_file_as_bytes;
use burn_store::{ModuleSnapshot, PytorchStore};
use burn::tensor::backend::Backend;
use dirs;
use crate::model::vgg::*;

fn get_cache_dir() -> PathBuf {
    let cache_dir = dirs::cache_dir()
        .expect("Failed to get cache directory models")
        .join("burn-models")
        .join("vgg");
    
    // Create cache directory for vgg models (in Burn) 
    // if it doesn't exist yet
    if !cache_dir.exists() {
        fs::create_dir_all(&cache_dir)
            .expect("Failed to create cache directory for VGG models in Burn");
    }
    
    cache_dir
}

fn download_pretrained_weights(cache_path: &PathBuf, vgg_version: &VggVersion) {
    if !cache_path.exists() {
        let url = get_vgg_url(vgg_version);
        let bytes = download_file_as_bytes(url, "Downloading VGG pretrained weights from PyTorch...");
        
        // Write to a temporary file. If writing was completed, then rename to the correct name.
        // If writing is not completed, the cache file with the correct name (i.e. `cache_path`) will
        // not exist so this code block will run again when this function gets called again.
        let temp_path = cache_path.with_extension("pth.temp");
        let mut file = File::create(&temp_path).expect("Failed to create a VGG model cache file");
        file.write_all(&bytes).expect("Failed to write pretrained VGG weights to the cache file");
        rename(temp_path, cache_path).expect("Failed to rename temporary file to the correct VGG19 cache file name");
    }
}

fn get_vgg_url(vgg_version: &VggVersion) -> &'static str {
    match vgg_version {
        VggVersion::Vgg11 => "https://download.pytorch.org/models/vgg11-8a719046.pth",
        VggVersion::Vgg11Bn => "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
        VggVersion::Vgg13 => "https://download.pytorch.org/models/vgg13-19584684.pth",
        VggVersion::Vgg13Bn => "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
        VggVersion::Vgg16 => "https://download.pytorch.org/models/vgg16-397923af.pth",
        VggVersion::Vgg16Bn => "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
        VggVersion::Vgg19 => "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
        VggVersion::Vgg19Bn => "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth"
    }
}

fn get_vgg_cache_file_name(vgg_version: &VggVersion) -> &'static str {
    match vgg_version {
        VggVersion::Vgg11 => "vgg11.pth",
        VggVersion::Vgg11Bn => "vgg11_bn.pth",
        VggVersion::Vgg13 => "vgg13.pth",
        VggVersion::Vgg13Bn => "vgg13_bn.pth",
        VggVersion::Vgg16 => "vgg16.pth",
        VggVersion::Vgg16Bn => "vgg16_bn.pth",
        VggVersion::Vgg19 => "vgg19.pth",
        VggVersion::Vgg19Bn => "vgg19_bn.pth"
    }
}

fn get_vgg_config_array(vgg_version: &VggVersion) -> &[i32] {
    match vgg_version {
        VggVersion::Vgg11 | VggVersion::Vgg11Bn => &[64, -1, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1],
        VggVersion::Vgg13 | VggVersion::Vgg13Bn => &[64, 64, -1, 128, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1],
        VggVersion::Vgg16 | VggVersion::Vgg16Bn => &[64, 64, -1, 128, 128, -1, 256, 256, 256, -1, 512, 512, 512, -1, 512, 512, 512, -1],
        VggVersion::Vgg19 | VggVersion::Vgg19Bn => &[64, 64, -1, 128, 128, -1, 256, 256, 256, 256, -1, 512, 512, 512, 512, -1, 512, 512, 512, 512, -1],
    }
}

fn is_batch_norm_vgg(vgg_version: &VggVersion) -> bool {
    match vgg_version {
        VggVersion::Vgg11Bn | VggVersion::Vgg13Bn | VggVersion::Vgg16Bn | VggVersion::Vgg19Bn => true,
        _ => false,
    }
}

fn generate_key_patterns(vgg_version: &VggVersion) -> Vec<(String, String)> {
    let vgg_config_array = get_vgg_config_array(&vgg_version);
    let is_batch_norm_vgg = is_batch_norm_vgg(&vgg_version);
    
    let mut key_patterns = Vec::new();
    
    let mut pytorch_index = 0;
    let mut burn_block_index = 1;
    let mut burn_layer_index = 0;
    
    for num in vgg_config_array {
        if *num == -1 {
            pytorch_index += 1;
            burn_block_index += 1;  // Only update
            burn_layer_index = 0;
        } else {
            key_patterns.push((
                format!("features.{}.weight", pytorch_index),
                format!("conv_block{}.conv_layers.{}.weight", burn_block_index, burn_layer_index)
            ));
            key_patterns.push((
                format!("features.{}.bias", pytorch_index),
                format!("conv_block{}.conv_layers.{}.bias", burn_block_index, burn_layer_index)
            ));
            
            if is_batch_norm_vgg {
                // Moving to batch_norm layer in the same block
                pytorch_index += 1;
                
                key_patterns.push((
                    format!("features.{}.weight", pytorch_index),
                    format!("conv_block{}.bn_layers.{}.gamma", burn_block_index, burn_layer_index)
                ));
                key_patterns.push((
                    format!("features.{}.bias", pytorch_index),
                    format!("conv_block{}.bn_layers.{}.beta", burn_block_index, burn_layer_index)
                )); 
                key_patterns.push((
                    format!("features.{}.running_mean", pytorch_index),
                    format!("conv_block{}.bn_layers.{}.running_mean", burn_block_index, burn_layer_index)
                ));
                key_patterns.push((
                    format!("features.{}.running_var", pytorch_index),
                    format!("conv_block{}.bn_layers.{}.running_var", burn_block_index, burn_layer_index)
                ));
            }
            
            pytorch_index += 2;
            burn_layer_index += 1;
        }
    }
    
    key_patterns
}

pub fn load_pretrained_weights<B: Backend>(mut vgg: Vgg<B>, vgg_version: VggVersion) -> Vgg<B> {
    let cache_dir = get_cache_dir();
    let cache_path = cache_dir.join(get_vgg_cache_file_name(&vgg_version));
    download_pretrained_weights(&cache_path, &vgg_version);
    
    let key_patterns = generate_key_patterns(&vgg_version);
    
    let mut store = PytorchStore::from_file(cache_path);
    for (pt_key, burn_key) in key_patterns {
        store = store.with_key_remapping(pt_key, burn_key);
    }
    
    let result = vgg.load_from(&mut store);
    if let Err(e) = result {
        eprintln!("Warning: Some weights for the VGG model could not be loaded: {:?}", e);
    }
    
    vgg
}