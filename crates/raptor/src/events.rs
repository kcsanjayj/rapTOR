//! Typed, append-only event log.
//!
//! This is the single source of truth for observability: the CLI timeline, JSONL trace files, and
//! the HTTP API are all *projections* of this stream (event sourcing).

use serde::Serialize;
use serde_json::Value;
use std::fs::File;
use std::io::Write;
use std::sync::Mutex;
use std::time::Instant;

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    RunStarted,
    MemoryRecalled,
    PlanRequested,
    PlanRejected,
    PlanFallback,
    PlanAccepted,
    NodeStarted,
    ToolCalled,
    AttemptFailed,
    ReviewStarted,
    CriticReview,
    CriticUnavailable,
    RepairRequested,
    NodeCompleted,
    NodeFailed,
    NodeSkipped,
    BudgetExhausted,
    SynthesisStarted,
    SynthesisFallback,
    RunFinished,
}

#[derive(Clone, Debug, Serialize)]
pub struct Event {
    pub seq: usize,
    /// Milliseconds since the run started.
    pub t_ms: u64,
    pub kind: EventKind,
    /// `engine`, `planner`, `worker-N`, `critic`, `synthesizer`.
    pub agent: String,
    pub node: Option<String>,
    pub msg: String,
    pub data: Value,
}

type Listener = Box<dyn Fn(&Event) + Send + Sync>;

pub struct EventLog {
    start: Instant,
    events: Mutex<Vec<Event>>,
    sink: Mutex<Option<File>>,
    listener: Mutex<Option<Listener>>,
}

impl Default for EventLog {
    fn default() -> Self {
        Self::new()
    }
}

impl EventLog {
    pub fn new() -> EventLog {
        EventLog {
            start: Instant::now(),
            events: Mutex::new(vec![]),
            sink: Mutex::new(None),
            listener: Mutex::new(None),
        }
    }

    /// Also append every event as one JSON line to `path` (best effort).
    pub fn with_trace_file(self, path: &std::path::Path) -> EventLog {
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(f) = File::create(path) {
            *self.sink.lock().unwrap_or_else(|e| e.into_inner()) = Some(f);
        }
        self
    }

    pub fn set_listener(&self, f: impl Fn(&Event) + Send + Sync + 'static) {
        *self.listener.lock().unwrap_or_else(|e| e.into_inner()) = Some(Box::new(f));
    }

    pub fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }

    pub fn emit(
        &self,
        kind: EventKind,
        agent: &str,
        node: Option<&str>,
        msg: impl Into<String>,
        data: Value,
    ) -> usize {
        let mut events = self.events.lock().unwrap_or_else(|e| e.into_inner());
        let ev = Event {
            seq: events.len(),
            t_ms: self.elapsed_ms(),
            kind,
            agent: agent.to_string(),
            node: node.map(|s| s.to_string()),
            msg: msg.into(),
            data,
        };
        if let Some(f) = self.sink.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
            if let Ok(line) = serde_json::to_string(&ev) {
                let _ = writeln!(f, "{line}");
            }
        }
        if let Some(l) = self
            .listener
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
        {
            l(&ev);
        }
        let seq = ev.seq;
        events.push(ev);
        seq
    }

    /// Events with `seq >= after`.
    pub fn since(&self, after: usize) -> Vec<Event> {
        let events = self.events.lock().unwrap_or_else(|e| e.into_inner());
        events.iter().skip(after).cloned().collect()
    }

    pub fn all(&self) -> Vec<Event> {
        self.since(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn sequence_numbers_are_dense_and_since_works() {
        let log = EventLog::new();
        for i in 0..5 {
            log.emit(
                EventKind::NodeStarted,
                "worker-1",
                Some("a"),
                format!("e{i}"),
                json!({}),
            );
        }
        let all = log.all();
        assert_eq!(
            all.iter().map(|e| e.seq).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4]
        );
        assert_eq!(log.since(3).len(), 2);
        assert_eq!(log.since(99).len(), 0);
    }

    #[test]
    fn concurrent_emit_keeps_seq_unique() {
        let log = std::sync::Arc::new(EventLog::new());
        let hs: Vec<_> = (0..8)
            .map(|_| {
                let l = log.clone();
                std::thread::spawn(move || {
                    for _ in 0..100 {
                        l.emit(EventKind::AttemptFailed, "w", None, "x", json!({}));
                    }
                })
            })
            .collect();
        for h in hs {
            h.join().unwrap();
        }
        let all = log.all();
        assert_eq!(all.len(), 800);
        assert!(all.windows(2).all(|w| w[1].seq == w[0].seq + 1));
    }
}
