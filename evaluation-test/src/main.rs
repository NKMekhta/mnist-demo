#![recursion_limit = "256"]

use std::path::PathBuf;

use burn::{
    data::dataset::{Dataset, vision::MnistDataset},
    module::Module as _,
    tensor::{ElementComparison, Tensor, cast::ToElement},
};
use training_lib::{Backend, FileRecorder, model::Model};

fn main() {
    let device = Default::default();

    let current_version: u32 = std::fs::read_to_string("model_storage/current")
        .expect("there should be a model version file")
        .trim()
        .parse()
        .expect("model version file is ill-formatted");
    
    let model_file = PathBuf::from("model_storage")
        .join(current_version.to_string())
        .join("model");

    let model = Model::<Backend>::new(&device)
        .load_file(model_file, &FileRecorder::default(), &device)
        .expect("model version file was not found");

    const ITEMS: usize = 1000;
    let dataset = MnistDataset::test();
    let result = dataset
        .iter()
        .take(ITEMS)
        .filter(|item| {
            let data = item.image.as_flattened();
            let data = Tensor::<Backend, 1>::from_floats(data, &device).reshape([1, 28, 28]);
            let data = ((data / 255) - 0.1307) / 0.3081;
            let output: Tensor<Backend, 2> = model.forward(data);
            let output = burn::tensor::activation::softmax(output, 1);
            let output = output
                .into_data()
                .iter::<f32>()
                .enumerate()
                .max_by(|p1, p2| p1.1.cmp(&p2.1))
                .expect("inference should have produced a proper tensor")
                .0;
            output.to_u8() == item.label
        })
        .count();

    println!("{}", (result as f64) / (ITEMS as f64))
}
