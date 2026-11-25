//! Budgeted LLM caller: shared by every agent in a run.
//!
//! * Enforces `max_llm_calls` across the whole run (cost / runaway-loop guard).
//! * Retries **transient** failures with exponential backoff; permanent failures fail fast.
//! * Counts calls and retries for the run metrics.

use crate::provider::{LlmError, LlmRequest, Provider};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

pub struct Llm {
    provider: Arc<dyn Provider>,
    max_calls: usize,
    retries: u32,
    backoff_base_ms: u64,
    calls: AtomicUsize,
    retried: AtomicUsize,
}

impl Llm {
    pub fn new(
        provider: Arc<dyn Provider>,
        max_calls: usize,
        retries: u32,
        backoff_base_ms: u64,
    ) -> Llm {
        Llm {
            provider,
            max_calls,
            retries,
            backoff_base_ms,
            calls: AtomicUsize::new(0),
            retried: AtomicUsize::new(0),
        }
    }

    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
    pub fn retries_used(&self) -> usize {
        self.retried.load(Ordering::SeqCst)
    }
    pub fn max_calls(&self) -> usize {
        self.max_calls
    }
    pub fn provider(&self) -> &Arc<dyn Provider> {
        &self.provider
    }

    /// `on_fail(attempt, error, will_retry)` is invoked for every failed attempt.
    pub fn call(
        &self,
        mut req: LlmRequest,
        on_fail: &dyn Fn(u32, &LlmError, bool),
    ) -> Result<String, LlmError> {
        for attempt in 1..=self.retries + 1 {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            if n >= self.max_calls {
                self.calls.fetch_sub(1, Ordering::SeqCst);
                return Err(LlmError {
                    transient: false,
                    budget: true,
                    msg: format!(
                        "LLM call budget exhausted (max_llm_calls={})",
                        self.max_calls
                    ),
                });
            }
            req.ctx.attempt = attempt;
            match self.provider.complete(&req) {
                Ok(text) => return Ok(text),
                Err(e) => {
                    let will_retry = e.transient && attempt <= self.retries;
                    on_fail(attempt, &e, will_retry);
                    if !will_retry {
                        return Err(e);
                    }
                    self.retried.fetch_add(1, Ordering::SeqCst);
                    let delay = (self.backoff_base_ms << (attempt - 1).min(5)).min(2000);
                    if delay > 0 {
                        std::thread::sleep(Duration::from_millis(delay));
                    }
                }
            }
        }
        Err(LlmError::permanent("retry loop exited unexpectedly"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{Ctx, Role};
    use std::sync::Mutex;

    struct Scripted(Mutex<Vec<Result<String, LlmError>>>);
    impl Provider for Scripted {
        fn name(&self) -> String {
            "scripted".into()
        }
        fn model(&self) -> String {
            "m".into()
        }
        fn complete(&self, _r: &LlmRequest) -> Result<String, LlmError> {
            self.0.lock().unwrap().remove(0)
        }
    }
    fn req() -> LlmRequest {
        LlmRequest {
            role: Role::Executor,
            system: String::new(),
            user: String::new(),
            ctx: Ctx::default(),
        }
    }
    fn llm(script: Vec<Result<String, LlmError>>, max: usize) -> Llm {
        Llm::new(Arc::new(Scripted(Mutex::new(script))), max, 2, 0)
    }

    #[test]
    fn retries_transient_then_succeeds() {
        let l = llm(
            vec![
                Err(LlmError::transient("503")),
                Err(LlmError::transient("503")),
                Ok("ok".into()),
            ],
            10,
        );
        assert_eq!(l.call(req(), &|_, _, _| {}).unwrap(), "ok");
        assert_eq!((l.calls(), l.retries_used()), (3, 2));
    }

    #[test]
    fn permanent_errors_are_not_retried() {
        let l = llm(vec![Err(LlmError::permanent("401"))], 10);
        assert!(l.call(req(), &|_, _, _| {}).is_err());
        assert_eq!(l.calls(), 1);
    }

    #[test]
    fn gives_up_after_retries_exhausted() {
        let l = llm(
            vec![
                Err(LlmError::transient("a")),
                Err(LlmError::transient("b")),
                Err(LlmError::transient("c")),
            ],
            10,
        );
        assert_eq!(l.call(req(), &|_, _, _| {}).unwrap_err().msg, "c");
        assert_eq!(l.calls(), 3);
    }

    #[test]
    fn budget_is_enforced_and_flagged() {
        let l = llm(vec![Ok("1".into()), Ok("2".into())], 2);
        l.call(req(), &|_, _, _| {}).unwrap();
        l.call(req(), &|_, _, _| {}).unwrap();
        let e = l.call(req(), &|_, _, _| {}).unwrap_err();
        assert!(e.budget && !e.transient, "{e:?}");
        assert_eq!(l.calls(), 2);
    }
}
