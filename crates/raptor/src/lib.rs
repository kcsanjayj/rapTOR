//! rapTOR core: a small, testable multi-agent orchestration engine.
//!
//! Pipeline: `plan -> validate DAG -> schedule (parallel) -> [execute -> critic -> repair]* -> synthesize`.
//! Every step emits a typed [`events::Event`] for the terminal timeline and optional HTTP API.

pub mod config;
pub mod critic;
pub mod dag;
pub mod engine;
pub mod events;
pub mod llm;
pub mod memory;
pub mod model;
pub mod planner;
pub mod provider;
pub mod server;
pub mod synthesizer;
pub mod tools;
pub mod util;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
