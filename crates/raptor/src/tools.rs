//! Deterministic tools the planner may schedule as DAG nodes.
//!
//! The calculator is a hand-written recursive-descent parser over `f64` with explicit limits
//! (length, nesting depth, exponent size, finite results). It fixes a v1 bug where `9**9**9`
//! could pin a CPU through Python big-integer exponentiation.

use serde_json::Value;

pub struct ToolSpec {
    pub name: &'static str,
    pub description: &'static str,
    pub args: &'static str,
}

pub struct Tools {
    specs: Vec<ToolSpec>,
}

impl Default for Tools {
    fn default() -> Self {
        Self::builtin()
    }
}

impl Tools {
    pub fn builtin() -> Tools {
        Tools {
            specs: vec![
                ToolSpec {
                    name: "calculator",
                    description: "Evaluate an arithmetic expression exactly (+ - * / % ^ ** and parentheses).",
                    args: r#"{"expression": "12*(3+4)"}"#,
                },
                ToolSpec {
                    name: "text_stats",
                    description: "Count characters, words and sentences in a piece of text.",
                    args: r#"{"text": "..."}"#,
                },
            ],
        }
    }

    pub fn names(&self) -> Vec<String> {
        self.specs.iter().map(|s| s.name.to_string()).collect()
    }

    /// Human/LLM readable catalogue, embedded in the planner prompt.
    pub fn describe(&self) -> String {
        self.specs
            .iter()
            .map(|s| format!("- {} : {} args={}", s.name, s.description, s.args))
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn execute(&self, name: &str, args: &Value) -> Result<String, String> {
        match name {
            "calculator" => {
                let expr = args
                    .get("expression")
                    .and_then(|v| v.as_str())
                    .ok_or("calculator requires a string argument 'expression'")?;
                eval_expression(expr).map(format_number)
            }
            "text_stats" => {
                let text = args
                    .get("text")
                    .and_then(|v| v.as_str())
                    .ok_or("text_stats requires a string argument 'text'")?;
                let words = text.split_whitespace().count();
                let sentences = text
                    .split(['.', '!', '?'])
                    .filter(|s| !s.trim().is_empty())
                    .count();
                Ok(format!(
                    "chars={} words={} sentences={}",
                    text.chars().count(),
                    words,
                    sentences
                ))
            }
            other => Err(format!("unknown tool '{other}'")),
        }
    }
}

/// Replace `{{task_id}}` placeholders in every string of `args` with that task's output.
pub fn substitute_args(args: &Value, deps: &[(String, String)]) -> Value {
    match args {
        Value::String(s) => {
            let mut out = s.clone();
            for (id, output) in deps {
                out = out.replace(&format!("{{{{{id}}}}}"), output.trim());
            }
            Value::String(out)
        }
        Value::Object(m) => Value::Object(
            m.iter()
                .map(|(k, v)| (k.clone(), substitute_args(v, deps)))
                .collect(),
        ),
        Value::Array(a) => Value::Array(a.iter().map(|v| substitute_args(v, deps)).collect()),
        other => other.clone(),
    }
}

pub fn format_number(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else if v.abs() >= 1e15 || (v != 0.0 && v.abs() < 1e-6) {
        format!("{v:e}")
    } else {
        let s = format!("{v:.10}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

// ---------------------------------------------------------------- calculator

const MAX_LEN: usize = 200;
const MAX_DEPTH: usize = 40;
const MAX_EXPONENT: f64 = 1024.0;

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Caret,
    LParen,
    RParen,
}

fn tokenize(src: &str) -> Result<Vec<Tok>, String> {
    let chars: Vec<char> = src.chars().collect();
    let mut i = 0;
    let mut out = vec![];
    while i < chars.len() {
        let c = chars[i];
        match c {
            ' ' | '\t' | '\n' => i += 1,
            '0'..='9' | '.' => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                    i += 1;
                }
                let s: String = chars[start..i].iter().collect();
                out.push(Tok::Num(
                    s.parse::<f64>()
                        .map_err(|_| format!("invalid number '{s}'"))?,
                ));
            }
            '+' => {
                out.push(Tok::Plus);
                i += 1
            }
            '-' => {
                out.push(Tok::Minus);
                i += 1
            }
            '*' => {
                if chars.get(i + 1) == Some(&'*') {
                    out.push(Tok::Caret);
                    i += 2;
                } else {
                    out.push(Tok::Star);
                    i += 1;
                }
            }
            '/' => {
                out.push(Tok::Slash);
                i += 1
            }
            '%' => {
                out.push(Tok::Percent);
                i += 1
            }
            '^' => {
                out.push(Tok::Caret);
                i += 1
            }
            '(' => {
                out.push(Tok::LParen);
                i += 1
            }
            ')' => {
                out.push(Tok::RParen);
                i += 1
            }
            other => return Err(format!("unexpected character '{other}'")),
        }
    }
    Ok(out)
}

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
    depth: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }
    fn enter(&mut self) -> Result<(), String> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err("expression is nested too deeply".into());
        }
        Ok(())
    }
    fn expr(&mut self) -> Result<f64, String> {
        let mut v = self.term()?;
        while let Some(t) = self.peek() {
            match t {
                Tok::Plus => {
                    self.pos += 1;
                    v += self.term()?;
                }
                Tok::Minus => {
                    self.pos += 1;
                    v -= self.term()?;
                }
                _ => break,
            }
        }
        Ok(v)
    }
    fn term(&mut self) -> Result<f64, String> {
        let mut v = self.unary()?;
        while let Some(t) = self.peek() {
            match t {
                Tok::Star => {
                    self.pos += 1;
                    v *= self.unary()?;
                }
                Tok::Slash => {
                    self.pos += 1;
                    let r = self.unary()?;
                    if r == 0.0 {
                        return Err("division by zero".into());
                    }
                    v /= r;
                }
                Tok::Percent => {
                    self.pos += 1;
                    let r = self.unary()?;
                    if r == 0.0 {
                        return Err("modulo by zero".into());
                    }
                    v %= r;
                }
                _ => break,
            }
        }
        Ok(v)
    }
    fn unary(&mut self) -> Result<f64, String> {
        self.enter()?;
        let r = match self.peek() {
            Some(Tok::Minus) => {
                self.pos += 1;
                self.unary().map(|v| -v)
            }
            Some(Tok::Plus) => {
                self.pos += 1;
                self.unary()
            }
            _ => self.power(),
        };
        self.depth -= 1;
        r
    }
    fn power(&mut self) -> Result<f64, String> {
        let base = self.atom()?;
        if let Some(Tok::Caret) = self.peek() {
            self.pos += 1;
            let exp = self.unary()?; // right-associative
            if exp.abs() > MAX_EXPONENT {
                return Err(format!(
                    "exponent too large (|exponent| must be <= {MAX_EXPONENT})"
                ));
            }
            let v = base.powf(exp);
            if !v.is_finite() {
                return Err("result is not a finite number".into());
            }
            return Ok(v);
        }
        Ok(base)
    }
    fn atom(&mut self) -> Result<f64, String> {
        match self.peek().cloned() {
            Some(Tok::Num(n)) => {
                self.pos += 1;
                Ok(n)
            }
            Some(Tok::LParen) => {
                self.pos += 1;
                self.enter()?;
                let v = self.expr()?;
                self.depth -= 1;
                match self.peek() {
                    Some(Tok::RParen) => {
                        self.pos += 1;
                        Ok(v)
                    }
                    _ => Err("missing closing parenthesis".into()),
                }
            }
            Some(t) => Err(format!("unexpected token {t:?}")),
            None => Err("unexpected end of expression".into()),
        }
    }
}

pub fn eval_expression(expr: &str) -> Result<f64, String> {
    if expr.chars().count() > MAX_LEN {
        return Err(format!("expression longer than {MAX_LEN} characters"));
    }
    let toks = tokenize(expr)?;
    if toks.is_empty() {
        return Err("empty expression".into());
    }
    let mut p = Parser {
        toks,
        pos: 0,
        depth: 0,
    };
    let v = p.expr()?;
    if p.pos != p.toks.len() {
        return Err("unexpected trailing input".into());
    }
    if !v.is_finite() {
        return Err("result is not a finite number".into());
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::Instant;

    fn ev(s: &str) -> String {
        format_number(eval_expression(s).unwrap())
    }

    #[test]
    fn arithmetic_and_precedence() {
        assert_eq!(ev("12*(3+4)"), "84");
        assert_eq!(ev("2+3*4"), "14");
        assert_eq!(ev("2^10"), "1024");
        assert_eq!(ev("2**3**2"), "512"); // right associative
        assert_eq!(ev("-2^2"), "-4");
        assert_eq!(ev("2**-1"), "0.5");
        assert_eq!(ev("10%4"), "2");
        assert_eq!(ev("1/3"), "0.3333333333");
    }

    #[test]
    fn rejects_dos_and_garbage_quickly() {
        let t = Instant::now();
        assert!(eval_expression("9**9**9").unwrap_err().contains("exponent"));
        assert!(eval_expression("10^10^10").is_err());
        assert!(eval_expression("1/0").unwrap_err().contains("zero"));
        assert!(eval_expression("(1+2").is_err());
        assert!(eval_expression("1+").is_err());
        assert!(eval_expression("").is_err());
        assert!(eval_expression("2 $ 3").is_err());
        assert!(eval_expression(&"(".repeat(100))
            .unwrap_err()
            .contains("deeply"));
        assert!(eval_expression(&"1+".repeat(150)).is_err()); // > 200 chars
        assert!(t.elapsed().as_millis() < 200, "must fail fast");
    }

    #[test]
    fn large_but_finite_results_are_formatted() {
        assert!(ev("2^1000").contains('e'));
        assert!(eval_expression("10^400").is_err()); // overflows f64 -> rejected, not inf
    }

    #[test]
    fn tool_dispatch_and_placeholders() {
        let t = Tools::builtin();
        assert_eq!(
            t.execute("calculator", &json!({"expression": "6*7"}))
                .unwrap(),
            "42"
        );
        assert!(t.execute("calculator", &json!({})).is_err());
        assert!(t.execute("nope", &json!({})).is_err());
        assert_eq!(
            t.execute("text_stats", &json!({"text": "Hi there. Bye!"}))
                .unwrap(),
            "chars=14 words=3 sentences=2"
        );
        let args = substitute_args(
            &json!({"expression": "{{calc}} + 1"}),
            &[("calc".into(), " 41 ".into())],
        );
        assert_eq!(args["expression"], "41 + 1");
    }
}
