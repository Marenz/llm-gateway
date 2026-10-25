//! HTTP body adapters. Provider state machines own assembly and terminal evidence.
use std::convert::Infallible;

use axum::body::Bytes;
use futures::{Stream, StreamExt};
use serde_json::Value;
use tokio::sync::mpsc;

use super::{chatgpt, sse, stream};

pub enum ProviderStream {
    Chatgpt(chatgpt::ChatgptStreamState),
    Anthropic(stream::StreamState),
}

impl ProviderStream {
    fn event(&mut self, event: &str, data: &str, model: &str, id: &str) -> Vec<String> {
        match self {
            Self::Chatgpt(state) => chatgpt::translate_stream_event(event, data, model, id, state),
            Self::Anthropic(state) => stream::translate_stream_event(data, model, id, state),
        }
    }

    fn fail(&mut self, error: String) -> Vec<String> {
        match self {
            Self::Chatgpt(state) => state.fail(error),
            Self::Anthropic(state) => state.fail(error),
        }
    }

    fn finish(&self) -> Result<(), String> {
        match self {
            Self::Chatgpt(state) => state.finish(),
            Self::Anthropic(state) => state.finish(),
        }
    }

    fn failed(&self) -> bool {
        match self {
            Self::Chatgpt(state) => state.failed(),
            Self::Anthropic(state) => state.failed(),
        }
    }
}

pub async fn collect_chatgpt(
    response: reqwest::Response,
    model: &str,
    request_id: &str,
) -> Result<Value, String> {
    let mut body = response.bytes_stream();
    let mut decoder = sse::Decoder::default();
    let mut state = chatgpt::ChatgptStreamState::default();
    while let Some(bytes) = body.next().await {
        let bytes = bytes.map_err(|_| "ChatGPT upstream transport failed".to_string())?;
        for (event, data) in decoder.push(&bytes)? {
            chatgpt::translate_stream_event(&event, &data, model, request_id, &mut state);
            if state.failed() {
                state.finish()?;
            }
        }
    }
    decoder.finish()?;
    state.completion(model, request_id)
}

pub fn translated_stream(
    response: reqwest::Response,
    model: String,
    request_id: String,
    mut state: ProviderStream,
) -> impl Stream<Item = Result<Bytes, Infallible>> {
    let (tx, rx) = mpsc::channel::<Bytes>(32);
    tokio::spawn(async move {
        let mut upstream = response.bytes_stream();
        let mut decoder = sse::Decoder::default();
        let result: Result<(), String> = async {
            while let Some(bytes) = upstream.next().await {
                let bytes = bytes.map_err(|_| "upstream SSE transport failed".to_string())?;
                for (event, data) in decoder.push(&bytes)? {
                    for chunk in state.event(&event, &data, &model, &request_id) {
                        if tx.send(Bytes::from(chunk)).await.is_err() {
                            return Ok(());
                        }
                    }
                    if state.failed() {
                        return Ok(());
                    }
                }
            }
            decoder.finish()?;
            state.finish()
        }
        .await;
        if let Err(error) = result {
            for chunk in state.fail(error) {
                if tx.send(Bytes::from(chunk)).await.is_err() {
                    break;
                }
            }
        }
    });
    futures::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|bytes| (Ok(bytes), rx))
    })
}
