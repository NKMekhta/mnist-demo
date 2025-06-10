use burn::{backend::{Autodiff, Wgpu}, record::DefaultRecorder};

pub mod config;
pub mod data;
pub mod model;
pub mod training;

pub type Backend = Autodiff<Wgpu>;
pub type FileRecorder = DefaultRecorder;