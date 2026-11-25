//! Small dependency-free helpers.

use std::time::{SystemTime, UNIX_EPOCH};

pub fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Deterministic pseudo-random number in `[0, 1)` from `(seed, key, salt)`.
/// FNV-1a over the key, then a splitmix64 finalizer. Used for reproducible fault injection.
pub fn hash_unit(seed: u64, key: &str, salt: u64) -> f64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ seed;
    for b in key.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h ^= salt.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut z = h.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// Truncate to `max` characters (not bytes), appending an ellipsis when cut.
pub fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut t: String = s.chars().take(max.saturating_sub(1)).collect();
    t.push('…');
    t
}

/// Return the first balanced `{ ... }` JSON object found in `text` (string/escape aware).
/// LLMs wrap JSON in prose or code fences; this recovers it without a regex.
pub fn extract_json_object(text: &str) -> Option<&str> {
    let bytes = text.as_bytes();
    let start = bytes.iter().position(|&b| b == b'{')?;
    let (mut depth, mut in_str, mut esc) = (0usize, false, false);
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_str {
            if esc {
                esc = false;
            } else if b == b'\\' {
                esc = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&text[start..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_deterministic_and_in_range() {
        let a = hash_unit(7, "task", 1);
        assert_eq!(a, hash_unit(7, "task", 1));
        assert_ne!(a, hash_unit(8, "task", 1));
        for i in 0..1000 {
            let v = hash_unit(1, "k", i);
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn truncate_counts_chars_not_bytes() {
        assert_eq!(truncate("héllo wörld", 5), "héll…");
        assert_eq!(truncate("short", 10), "short");
    }

    #[test]
    fn json_extraction_handles_fences_and_braces_in_strings() {
        let t = "Sure!\n```json\n{\"a\": \"}{\", \"b\": {\"c\": 1}}\n```\nbye";
        assert_eq!(
            extract_json_object(t),
            Some("{\"a\": \"}{\", \"b\": {\"c\": 1}}")
        );
        assert_eq!(extract_json_object("no json here"), None);
        assert_eq!(extract_json_object("{ unbalanced"), None);
    }
}
