#![recursion_limit = "256"]

use burn::{backend::wgpu::WgpuDevice, module::Module as _, tensor::Tensor};
use futures_util::{SinkExt, StreamExt};
use std::{
    io::Error,
    sync::{Arc, Mutex},
};
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::tungstenite::{Bytes, Message};
use training_lib::{Backend, FileRecorder, model::Model};

const MODEL_FILE: &str = "./model";
const ADDRESS: &str = "127.0.0.1:8080";

#[tokio::main]
async fn main() -> Result<(), Error> {
    let device = Default::default();

    let model = Model::<Backend>::new(&device)
        .load_file(&MODEL_FILE, &FileRecorder::default(), &device)
        .expect("model version file was not found");

    let model: Arc<Mutex<_>> = Arc::new(Mutex::new(model));
    let device: Arc<Mutex<_>> = Arc::new(Mutex::new(device));

    let listener = TcpListener::bind(&ADDRESS).await.expect("Failed to bind");

    while let Ok((stream, _)) = listener.accept().await {
        tokio::spawn(accept_connection(stream, model.clone(), device.clone()));
    }

    Ok(())
}

async fn accept_connection(
    stream: TcpStream,
    model: Arc<Mutex<Model<Backend>>>,
    device: Arc<Mutex<WgpuDevice>>,
) {
    let ws_stream = tokio_tungstenite::accept_async(stream)
        .await
        .expect("Error during the websocket handshake occurred");

    let (mut write, mut read) = ws_stream.split();

    while let Some(msg) = read.next().await {
        let Ok(msg) = msg else {
            continue;
        };

        let Message::Binary(msg) = msg else {
            continue;
        };

        if msg.len() != std::mem::size_of::<f32>() * 28 * 28 {
            continue;
        }

        let floats: Vec<f32> = msg
            .chunks(std::mem::size_of::<f32>())
            .map(|b| f32::from_be_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        let data = Tensor::<Backend, 1>::from_floats(floats.as_slice(), &*device.lock().unwrap())
            .reshape([1, 28, 28]);
        let data = ((data / 255) - 0.1307) / 0.3081;
        let output: Tensor<Backend, 2> = model.lock().unwrap().forward(data);
        let output = burn::tensor::activation::softmax(output, 1);
        let output = output
            .into_data()
            .iter::<f32>()
            .flat_map(|f| f.to_be_bytes())
            .collect::<Bytes>();

        write.send(Message::Binary(output)).await.unwrap();
    }
}
