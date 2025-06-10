use std::num::NonZeroUsize;
use std::path::PathBuf;
use burn::config::Config;
use burn::optim::decay::WeightDecayConfig;
use burn::optim::AdamConfig;
use burn::tensor::cast::ToElement;

#[derive(Config)]
pub struct TrainingConfig {
    pub num_epochs: u32,
    pub batch_size: u32,
    pub num_workers: u16,
    pub seed: u64,
    pub model_storage: PathBuf,
    pub initializer_version: Option<u32>,
    pub output_version: u32,
    pub optimizer_config: AdamConfig,
}

impl TrainingConfig {
    pub fn with_artifact_location(self, model_storage: PathBuf) -> Self {
        Self { model_storage, ..self }
    }

    pub fn with_output_version(self, output_version: u32) -> Self {
        Self { output_version, ..self }
    }

    pub fn with_loaded_version(self, version: u32) -> Self {
        Self { initializer_version: Some(version), ..self }
    }

    pub fn with_epochs(self, num_epochs: u32) -> Self {
        Self { num_epochs, ..self }
    }

    pub fn with_batch_size(self, batch_size: u32) -> Self {
        Self { batch_size, ..self }
    }

    pub fn with_worker_threads(self, num_workers: u16) -> Self {
        Self { num_workers, ..self }
    }

    pub fn with_randomizer_seed(self, seed: u64) -> Self {
        Self { seed, ..self }
    }
    
    pub fn with_optimizer_config(self, optimizer_config: AdamConfig) -> Self {
        Self { optimizer_config, ..self }
    }

    pub fn destination_path(&self) -> PathBuf {
        self.model_storage.join(self.output_version.to_string())
    }
    
    pub fn output_model_dump(&self) -> PathBuf {
        self.destination_path().join("model")
    }

    pub fn loaded_model_dump(&self) -> Option<PathBuf> {
        Some(
            self.model_storage
                .join(self.initializer_version?.to_string())
                .join("model")
        )
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        let num_workers = std::thread::available_parallelism()
            .map(NonZeroUsize::get)
            .unwrap_or(4)
            .to_u16();
        let optimizer_config = AdamConfig::new()
            .with_weight_decay(Some(WeightDecayConfig::new(5e-5)));
        
        Self {
            num_epochs: 10,
            batch_size: 64,
            num_workers,
            seed: rand::random(),
            model_storage: "../model_storage".into(),
            output_version: 1,
            initializer_version: None,
            optimizer_config,
        }
    }
}
