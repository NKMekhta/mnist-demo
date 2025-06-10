use training_lib::{
    config::TrainingConfig, training,
    Backend, FileRecorder
};

fn main() {
    let train = training::run::<Backend, FileRecorder>;

    let device = Default::default();
    let config = TrainingConfig::default()
        .with_artifact_location("../model_storage".into())
        .with_output_version(1)
        .with_epochs(1)
        .with_batch_size(128)
        .with_worker_threads(8)
        .with_randomizer_seed(15);

    train(device, &config);
}
