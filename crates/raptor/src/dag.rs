//! Plan validation and graph algorithms.
//!
//! Design rule (fixes a v1 bug): ids and dependency references go through the *same*
//! normalizer, unknown dependencies are an **error** (never silently dropped), and cycles
//! are rejected before anything executes.

use crate::model::{Kind, Plan};
use crate::util::truncate;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanError(pub String);

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for PlanError {}

#[derive(Clone, Debug)]
pub struct PlanLimits {
    pub max_tasks: usize,
}

impl Default for PlanLimits {
    fn default() -> Self {
        PlanLimits { max_tasks: 12 }
    }
}

/// `"Research Python!"` -> `"research_python"`. ASCII alphanumerics only, max 48 chars.
pub fn normalize_id(raw: &str) -> String {
    let mut out = String::new();
    let mut last_underscore = true;
    for c in raw.trim().chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            last_underscore = false;
        } else if !last_underscore {
            out.push('_');
            last_underscore = true;
        }
    }
    while out.ends_with('_') {
        out.pop();
    }
    out.chars().take(48).collect()
}

/// Validate and normalize a plan in place.
pub fn validate(
    plan: &mut Plan,
    limits: &PlanLimits,
    known_tools: &[String],
) -> Result<(), PlanError> {
    if plan.tasks.is_empty() {
        return Err(PlanError("plan has no tasks".into()));
    }
    if plan.tasks.len() > limits.max_tasks {
        return Err(PlanError(format!(
            "plan has {} tasks; the limit is {}",
            plan.tasks.len(),
            limits.max_tasks
        )));
    }

    let mut ids: HashSet<String> = HashSet::new();
    for t in plan.tasks.iter_mut() {
        let original = t.id.clone();
        t.id = normalize_id(&t.id);
        if t.id.is_empty() {
            return Err(PlanError(format!(
                "task id {original:?} is empty after normalization"
            )));
        }
        if !ids.insert(t.id.clone()) {
            return Err(PlanError(format!("duplicate task id '{}'", t.id)));
        }
        t.description = truncate(t.description.trim(), 400);
        if t.description.is_empty() {
            return Err(PlanError(format!(
                "task '{}' has an empty description",
                t.id
            )));
        }
        match t.kind {
            Kind::Tool => {
                let name = t.tool.clone().unwrap_or_default();
                if name.is_empty() {
                    return Err(PlanError(format!("tool task '{}' has no tool name", t.id)));
                }
                if !known_tools.iter().any(|k| k == &name) {
                    return Err(PlanError(format!(
                        "task '{}' uses unknown tool '{}' (available: {})",
                        t.id,
                        name,
                        known_tools.join(", ")
                    )));
                }
                if t.args.is_null() {
                    t.args = serde_json::json!({});
                }
                if !t.args.is_object() {
                    return Err(PlanError(format!(
                        "tool task '{}' args must be a JSON object",
                        t.id
                    )));
                }
            }
            Kind::Llm => {
                t.tool = None;
                t.args = serde_json::Value::Null;
            }
        }
    }

    for t in plan.tasks.iter_mut() {
        let mut seen = HashSet::new();
        let mut deps = Vec::new();
        for raw in t.dependencies.drain(..) {
            let d = normalize_id(&raw);
            if d == t.id {
                return Err(PlanError(format!("task '{}' depends on itself", t.id)));
            }
            if !ids.contains(&d) {
                return Err(PlanError(format!(
                    "task '{}' depends on unknown task '{}' (known: {})",
                    t.id,
                    raw,
                    ids.iter().cloned().collect::<Vec<_>>().join(", ")
                )));
            }
            if seen.insert(d.clone()) {
                deps.push(d);
            }
        }
        t.dependencies = deps;
    }

    let stuck = cycle_members(plan);
    if !stuck.is_empty() {
        return Err(PlanError(format!(
            "dependency cycle detected among or downstream of: {}",
            stuck.join(", ")
        )));
    }
    Ok(())
}

/// Kahn's algorithm. Returns the tasks that can never be scheduled (empty => acyclic).
pub fn cycle_members(plan: &Plan) -> Vec<String> {
    let mut indeg: HashMap<&str, usize> = plan
        .tasks
        .iter()
        .map(|t| (t.id.as_str(), t.dependencies.len()))
        .collect();
    let mut queue: VecDeque<&str> = plan
        .tasks
        .iter()
        .filter(|t| t.dependencies.is_empty())
        .map(|t| t.id.as_str())
        .collect();
    let mut done = 0usize;
    while let Some(id) = queue.pop_front() {
        done += 1;
        for t in &plan.tasks {
            if t.dependencies.iter().any(|d| d == id) {
                let e = indeg.get_mut(t.id.as_str()).unwrap();
                *e -= 1;
                if *e == 0 {
                    queue.push_back(t.id.as_str());
                }
            }
        }
    }
    if done == plan.tasks.len() {
        return vec![];
    }
    plan.tasks
        .iter()
        .filter(|t| indeg[t.id.as_str()] > 0)
        .map(|t| t.id.clone())
        .collect()
}

/// Topological layers (longest-path depth). Layer 0 are the roots.
pub fn levels(plan: &Plan) -> Vec<Vec<String>> {
    let mut depth: HashMap<&str, usize> = HashMap::new();
    let by_id: HashMap<&str, &crate::model::Task> =
        plan.tasks.iter().map(|t| (t.id.as_str(), t)).collect();
    fn d<'a>(
        id: &'a str,
        by_id: &HashMap<&'a str, &'a crate::model::Task>,
        memo: &mut HashMap<&'a str, usize>,
    ) -> usize {
        if let Some(v) = memo.get(id) {
            return *v;
        }
        let t = by_id[id];
        let v = t
            .dependencies
            .iter()
            .map(|x| d(x.as_str(), by_id, memo) + 1)
            .max()
            .unwrap_or(0);
        memo.insert(id, v);
        v
    }
    let mut out: Vec<Vec<String>> = Vec::new();
    for t in &plan.tasks {
        let v = d(t.id.as_str(), &by_id, &mut depth);
        if out.len() <= v {
            out.resize(v + 1, vec![]);
        }
        out[v].push(t.id.clone());
    }
    out
}

/// All tasks that (transitively) depend on `id`, in plan order.
pub fn transitive_dependents(plan: &Plan, id: &str) -> Vec<String> {
    let mut hit: HashSet<String> = HashSet::new();
    let mut frontier = vec![id.to_string()];
    while let Some(cur) = frontier.pop() {
        for t in &plan.tasks {
            if t.dependencies.iter().any(|d| d == &cur) && hit.insert(t.id.clone()) {
                frontier.push(t.id.clone());
            }
        }
    }
    plan.tasks
        .iter()
        .filter(|t| hit.contains(&t.id))
        .map(|t| t.id.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Task;

    fn plan(tasks: Vec<Task>) -> Plan {
        Plan {
            goal: "g".into(),
            tasks,
        }
    }
    fn tools() -> Vec<String> {
        vec!["calculator".into()]
    }

    #[test]
    fn normalizes_ids_consistently_for_ids_and_deps() {
        let mut p = plan(vec![
            Task::llm("Research Python!", "a", &[]),
            Task::llm("compare", "b", &["research python"]),
        ]);
        validate(&mut p, &PlanLimits::default(), &tools()).unwrap();
        assert_eq!(p.tasks[0].id, "research_python");
        assert_eq!(p.tasks[1].dependencies, vec!["research_python"]);
    }

    #[test]
    fn rejects_cycles() {
        let mut p = plan(vec![
            Task::llm("a", "x", &["b"]),
            Task::llm("b", "y", &["a"]),
        ]);
        let e = validate(&mut p, &PlanLimits::default(), &tools()).unwrap_err();
        assert!(e.0.contains("cycle"), "{e}");
    }

    #[test]
    fn rejects_unknown_dependency_instead_of_dropping_it() {
        let mut p = plan(vec![Task::llm("a", "x", &["ghost"])]);
        let e = validate(&mut p, &PlanLimits::default(), &tools()).unwrap_err();
        assert!(e.0.contains("unknown task"), "{e}");
    }

    #[test]
    fn rejects_self_dependency_duplicates_empty_and_oversize() {
        let l = PlanLimits { max_tasks: 2 };
        assert!(validate(&mut plan(vec![Task::llm("a", "x", &["a"])]), &l, &tools()).is_err());
        assert!(validate(
            &mut plan(vec![Task::llm("a", "x", &[]), Task::llm("A", "y", &[])]),
            &l,
            &tools()
        )
        .is_err());
        assert!(validate(&mut plan(vec![]), &l, &tools()).is_err());
        assert!(validate(&mut plan(vec![Task::llm("a", " ", &[])]), &l, &tools()).is_err());
        let three = plan(vec![
            Task::llm("a", "x", &[]),
            Task::llm("b", "x", &[]),
            Task::llm("c", "x", &[]),
        ]);
        assert!(validate(&mut three.clone(), &l, &tools())
            .unwrap_err()
            .0
            .contains("limit"));
    }

    #[test]
    fn rejects_unknown_tool() {
        let mut t = Task::llm("a", "x", &[]);
        t.kind = Kind::Tool;
        t.tool = Some("rm_rf".into());
        let e = validate(&mut plan(vec![t]), &PlanLimits::default(), &tools()).unwrap_err();
        assert!(e.0.contains("unknown tool"), "{e}");
    }

    #[test]
    fn levels_and_dependents() {
        let mut p = plan(vec![
            Task::llm("a", "x", &[]),
            Task::llm("b", "x", &[]),
            Task::llm("c", "x", &["a", "b"]),
            Task::llm("d", "x", &["c"]),
        ]);
        validate(&mut p, &PlanLimits::default(), &tools()).unwrap();
        assert_eq!(levels(&p), vec![vec!["a", "b"], vec!["c"], vec!["d"]]);
        assert_eq!(transitive_dependents(&p, "a"), vec!["c", "d"]);
        assert!(transitive_dependents(&p, "d").is_empty());
    }
}
