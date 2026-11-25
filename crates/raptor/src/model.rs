//! Core data types shared by planner, scheduler, UI and API.

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    #[default]
    Llm,
    Tool,
}

/// One node of the plan DAG.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    #[serde(alias = "task")]
    pub description: String,
    #[serde(default, alias = "deps", alias = "depends_on")]
    pub dependencies: Vec<String>,
    #[serde(default)]
    pub kind: Kind,
    /// Tool name when `kind == Tool`.
    #[serde(default)]
    pub tool: Option<String>,
    /// Tool arguments. String values may contain `{{task_id}}` placeholders that are
    /// replaced with that upstream task's output before the tool runs.
    #[serde(default)]
    pub args: Value,
}

impl Task {
    pub fn llm(id: &str, description: &str, deps: &[&str]) -> Task {
        Task {
            id: id.into(),
            description: description.into(),
            dependencies: deps.iter().map(|s| s.to_string()).collect(),
            kind: Kind::Llm,
            tool: None,
            args: Value::Null,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Plan {
    #[serde(default)]
    pub goal: String,
    pub tasks: Vec<Task>,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NodeStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Skipped,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    /// Every task completed.
    Succeeded,
    /// At least one task completed, at least one did not.
    Partial,
    /// Nothing usable was produced.
    Failed,
}
