//! Small persistent memory: past (request, answer) pairs recalled by token overlap.
//!
//! Honest scope: this is lexical Jaccard similarity over a JSONL file, not embeddings.
//! It is *wired in* (recalled before planning, saved after success), tested, and bounded.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

use crate::util::{truncate, unix_ms};

const MAX_RECORDS: usize = 500;
const MIN_SCORE: f64 = 0.25;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub ts_ms: u64,
    pub request: String,
    pub answer: String,
}

pub struct MemoryStore {
    path: PathBuf,
    records: Mutex<Vec<MemoryRecord>>,
}

const STOP: &[&str] = &[
    "the", "and", "for", "with", "that", "this", "what", "which", "who", "how", "are", "was",
    "you", "your", "from", "about", "into", "can", "should", "would", "could", "better", "best",
    "please", "tell", "give",
];

fn tokens(s: &str) -> HashSet<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= 3 && !STOP.contains(w))
        .map(|w| w.to_string())
        .collect()
}

pub fn similarity(a: &str, b: &str) -> f64 {
    let (ta, tb) = (tokens(a), tokens(b));
    if ta.is_empty() || tb.is_empty() {
        return 0.0;
    }
    let inter = ta.intersection(&tb).count() as f64;
    let union = ta.union(&tb).count() as f64;
    inter / union
}

impl MemoryStore {
    pub fn open(dir: &std::path::Path) -> MemoryStore {
        let _ = fs::create_dir_all(dir);
        let path = dir.join("memory.jsonl");
        let mut records: Vec<MemoryRecord> = fs::read_to_string(&path)
            .unwrap_or_default()
            .lines()
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect();
        if records.len() > MAX_RECORDS {
            records.drain(..records.len() - MAX_RECORDS);
        }
        MemoryStore {
            path,
            records: Mutex::new(records),
        }
    }

    pub fn len(&self) -> usize {
        self.records.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn save(&self, request: &str, answer: &str) {
        let rec = MemoryRecord {
            ts_ms: unix_ms(),
            request: truncate(request, 500),
            answer: truncate(answer, 1200),
        };
        if let Ok(mut f) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
        {
            if let Ok(line) = serde_json::to_string(&rec) {
                let _ = writeln!(f, "{line}");
            }
        }
        let mut r = self.records.lock().unwrap();
        r.push(rec);
        if r.len() > MAX_RECORDS {
            r.remove(0);
        }
    }

    /// Top-`k` records whose request overlaps `query` (score >= 0.25), best first.
    pub fn recall(&self, query: &str, k: usize) -> Vec<MemoryRecord> {
        let r = self.records.lock().unwrap();
        let mut scored: Vec<(f64, &MemoryRecord)> = r
            .iter()
            .map(|rec| (similarity(query, &rec.request), rec))
            .filter(|(s, _)| *s >= MIN_SCORE)
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored
            .into_iter()
            .take(k)
            .map(|(_, rec)| rec.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("raptor-mem-{}-{}", name, unix_ms()));
        let _ = fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn recalls_similar_and_ignores_unrelated() {
        let d = tmp("recall");
        let m = MemoryStore::open(&d);
        m.save(
            "Compare Rust and Go for a CLI tool",
            "Go is simpler, Rust is faster.",
        );
        m.save("Plan a 3-day trip to Rome", "Day 1: Colosseum.");
        let hits = m.recall("Compare Rust and Zig for a CLI tool", 3);
        assert_eq!(hits.len(), 1);
        assert!(hits[0].request.contains("Rust and Go"));
        assert!(m.recall("quantum chromodynamics lecture", 3).is_empty());
    }

    #[test]
    fn persists_across_reopen_and_is_bounded() {
        let d = tmp("persist");
        {
            let m = MemoryStore::open(&d);
            for i in 0..(MAX_RECORDS + 20) {
                m.save(&format!("request number {i} alpha"), "a");
            }
            assert_eq!(m.len(), MAX_RECORDS);
        }
        let m2 = MemoryStore::open(&d);
        assert_eq!(m2.len(), MAX_RECORDS);
    }
}
