use base64::engine::general_purpose::{STANDARD, STANDARD_NO_PAD};
use base64::Engine;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub const PROGRAM_RESUME_TOKEN_ENV: &str = "OSC7577_TOKEN";

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct ProgramResumeRecord {
    pub argv: Vec<String>,
    pub cwd: Option<String>,
}

#[derive(Default)]
pub(crate) struct ProgramResume {
    token: Option<String>,
    records: HashMap<u32, (u64, Option<ProgramResumeRecord>)>,
}

impl ProgramResume {
    pub fn set_token(&mut self, token: String) {
        self.token = Some(token);
        self.records.clear();
    }

    pub fn record(&self, pid: u32) -> Option<ProgramResumeRecord> {
        self.records.get(&pid)?.1.clone()
    }

    pub fn apply(&mut self, body: &[u8]) {
        if let Ok(now) = SystemTime::now().duration_since(UNIX_EPOCH) {
            self.apply_at(body, now.as_secs());
        }
    }

    fn apply_at(&mut self, body: &[u8], now: u64) {
        let Some(token) = self.token.as_deref() else {
            return;
        };
        if body.len() > 4096 {
            return;
        }
        let Ok(body) = std::str::from_utf8(body) else {
            return;
        };
        let Some((payload, mac)) = body.rsplit_once(":mac=") else {
            return;
        };
        if mac.len() != 64
            || !mac
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return;
        }
        let mut hash = Sha256::new();
        hash.update(payload);
        hash.update(b":");
        hash.update(token);
        let expected = hex::encode(hash.finalize());
        if mac
            .bytes()
            .zip(expected.bytes())
            .fold(0, |diff, (a, b)| diff | (a ^ b))
            != 0
        {
            return;
        }
        let Some(time) = field(payload, "time").and_then(positive::<u64>) else {
            return;
        };
        if now.abs_diff(time) > 60 {
            return;
        }
        let Some(pid) = field(payload, "pid").and_then(positive::<u32>) else {
            return;
        };
        if self.records.get(&pid).is_some_and(|(last, _)| time < *last) {
            return;
        }
        self.records
            .retain(|_, (last, record)| record.is_some() || now.saturating_sub(*last) <= 60);
        if !self.records.contains_key(&pid) && self.records.len() >= 64 {
            return;
        }
        // Authenticated invalid reports clear the owner's previous command.
        self.records.insert(pid, (time, parse_record(payload)));
    }
}

fn field<'a>(payload: &'a str, key: &str) -> Option<&'a str> {
    let mut matches = payload
        .split(':')
        .filter_map(|pair| pair.split_once('='))
        .filter_map(|(name, value)| (name == key).then_some(value));
    let value = matches.next()?;
    matches.next().is_none().then_some(value)
}

fn positive<T: std::str::FromStr>(value: &str) -> Option<T> {
    if value.starts_with('0') || !value.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    value.parse().ok()
}

fn decode(value: &str) -> Option<String> {
    let decoded = if value.ends_with('=') {
        STANDARD.decode(value)
    } else {
        STANDARD_NO_PAD.decode(value)
    };
    let text = String::from_utf8(decoded.ok()?).ok()?;
    (!text.chars().any(char::is_control)).then_some(text)
}

fn parse_record(payload: &str) -> Option<ProgramResumeRecord> {
    let mut seen = HashSet::new();
    for pair in payload.split(':') {
        let (key, value) = pair.split_once('=')?;
        if key.is_empty()
            || !key.bytes().all(|b| b.is_ascii_lowercase())
            || !value
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"_.,+/=-".contains(&b))
            || key == "mac"
            || !seen.insert(key)
        {
            return None;
        }
    }
    if let Some(need) = field(payload, "need") {
        if !need.split(',').all(|key| {
            matches!(
                key,
                "pid" | "time" | "argv" | "cwd" | "clear" | "need" | "mac"
            )
        }) {
            return None;
        }
    }
    if seen.contains("clear") {
        return None;
    }
    let argv: Vec<String> = field(payload, "argv")?
        .split(',')
        .map(decode)
        .collect::<Option<_>>()?;
    let command = argv.first()?;
    if argv.len() > 128
        || command.is_empty()
        || (!Path::new(command).is_absolute()
            && (command.contains('/') || command.starts_with('-')))
    {
        return None;
    }
    let cwd = if let Some(value) = field(payload, "cwd") {
        let value = decode(value)?;
        if !Path::new(&value).is_absolute() {
            return None;
        }
        Some(value)
    } else {
        None
    };
    Some(ProgramResumeRecord { argv, cwd })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOKEN: &str = "dummy-token-replace-me";
    const NOW: u64 = 1795543200;

    fn report(payload: &str) -> String {
        let hash = Sha256::digest(format!("{payload}:{TOKEN}"));
        format!("{payload}:mac={}", hex::encode(hash))
    }

    #[test]
    fn verifies_documented_vector_and_clears() {
        let mut resume = ProgramResume::default();
        resume.set_token(TOKEN.to_owned());
        resume.apply_at(b"pid=74877:time=1795543200:argv=bnZpbQ==,LVM=,L2hvbWUvdXNlci8ubG9jYWwvc3RhdGUvbnZpbS9zZXNzaW9ucy84ZjJjMWQudmlt:mac=9dbb99bfb856749a87563839d7ce9062818d78cf25909db93f309e0cd3b1909b", NOW);
        assert_eq!(
            resume.record(74877).unwrap().argv,
            [
                "nvim",
                "-S",
                "/home/user/.local/state/nvim/sessions/8f2c1d.vim"
            ],
        );
        resume.apply_at(b"pid=74877:time=1795543200:clear=1:mac=c30603369f98fef581a4098be889175b0c74f0641e49f8b25f4326cb38f97c93", NOW);
        assert!(resume.record(74877).is_none());
    }

    #[test]
    fn ignores_unauthenticated_stale_and_out_of_order_reports() {
        let mut resume = ProgramResume::default();
        let first = report("pid=42:time=1795543200:argv=eA==,");
        resume.apply_at(first.as_bytes(), NOW);
        assert!(resume.record(42).is_none());
        resume.set_token(TOKEN.to_owned());
        resume.apply_at(first.as_bytes(), NOW);
        for invalid in [
            "pid=42:time=1795543200:clear=1:mac=bad".to_owned(),
            report("pid=42:time=1795543199:clear=1"),
            report("pid=42:time=1795543299:clear=1"),
        ] {
            resume.apply_at(invalid.as_bytes(), NOW);
        }
        assert_eq!(resume.record(42).unwrap().argv, ["x", ""]);

        let invalid = report("pid=42:time=1795543201:argv=eA==:need=unsupported");
        resume.apply_at(invalid.as_bytes(), NOW);
        assert!(resume.record(42).is_none());
        resume.apply_at(first.as_bytes(), NOW);
        assert!(resume.record(42).is_none());
    }
}
