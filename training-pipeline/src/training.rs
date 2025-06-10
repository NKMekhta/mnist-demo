use std::path::Path;

use crate::{config::TrainingConfig, data::MnistBatcher, model::Model};

use burn::{
    config::Config as _,
    data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
    module::Module,
    record::{CompactRecorder, FileRecorder},
    tensor::{backend::AutodiffBackend, cast::ToElement},
    train::{
        LearnerBuilder, MetricEarlyStoppingStrategy, StoppingCondition,
        metric::{
            AccuracyMetric, CpuMemory, CpuTemperature, CpuUse, LossMetric,
            store::{Aggregate, Direction, Split},
        },
    },
};

pub fn run<B: AutodiffBackend, FR: FileRecorder<B> + Default>(
    device: B::Device,
    config: &TrainingConfig,
) {
    create_artifact_dir(&config.destination_path());
    B::seed(config.seed);

    let model = Model::<B>::new(&device);
    let model = match config.loaded_model_dump() {
        Some(dump) => model
            .load_file(dump, &FR::default(), &device)
            .expect("model version file was not found"),
        None => model,
    };

    let batcher = MnistBatcher::default();

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size.to_usize())
        .shuffle(config.seed)
        .num_workers(config.num_workers.to_usize())
        .build(MnistDataset::train());

    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size.to_usize())
        .shuffle(config.seed)
        .num_workers(config.num_workers.to_usize())
        .build(MnistDataset::test());

    let learner = LearnerBuilder::new(config.destination_path())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(CpuUse::new())
        .metric_valid_numeric(CpuUse::new())
        .metric_train_numeric(CpuMemory::new())
        .metric_valid_numeric(CpuMemory::new())
        .metric_train_numeric(CpuTemperature::new())
        .metric_valid_numeric(CpuTemperature::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .early_stopping(MetricEarlyStoppingStrategy::new(
            &LossMetric::<B>::new(),
            Aggregate::Mean,
            Direction::Lowest,
            Split::Valid,
            StoppingCondition::NoImprovementSince { n_epochs: 1 },
        ))
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs.to_usize())
        .summary()
        .build(model, config.optimizer_config.init(), 1e-4);

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    config
        .save(config.destination_path().join("config.json"))
        .unwrap();

    model_trained
        .save_file(config.output_model_dump(), &FR::default())
        .expect("Failed to save trained model");
}

fn create_artifact_dir(artifact_dir: &Path) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}
