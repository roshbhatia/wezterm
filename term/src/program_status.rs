use base64::engine::general_purpose::{STANDARD, STANDARD_NO_PAD};
use base64::Engine;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProgramStatusState {
    Idle,
    Working,
    Done,
    Blocked,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgramStatusRecord {
    pub id: String,
    pub state: ProgramStatusState,
    pub kind: Option<String>,
    pub progress: Option<u8>,
    pub app: Option<String>,
    pub title: Option<String>,
    pub msg: Option<String>,
}

#[derive(Debug, Default)]
pub(crate) struct ProgramStatus {
    records: VecDeque<ProgramStatusRecord>,
}

fn valid_name(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 32
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_.+-".contains(&b))
}

fn decode_text(value: &str, encoded_limit: usize, decoded_limit: usize) -> Option<String> {
    if value.len() > encoded_limit {
        return None;
    }
    let bytes = STANDARD
        .decode(value)
        .or_else(|_| STANDARD_NO_PAD.decode(value))
        .ok()?;
    if bytes.len() > decoded_limit {
        return None;
    }
    let text = String::from_utf8(bytes).ok()?;
    if text.chars().any(char::is_control) {
        return None;
    }
    Some(text)
}

impl ProgramStatus {
    pub fn records(&self) -> Vec<ProgramStatusRecord> {
        self.records
            .iter()
            .map(|record| {
                let mut result = record.clone();
                if result.app.is_none() {
                    let mut id = result.id.as_str();
                    while !id.is_empty() {
                        id = id.rsplit_once('/').map(|(parent, _)| parent).unwrap_or("");
                        if let Some(app) = self
                            .records
                            .iter()
                            .find(|r| r.id == id)
                            .and_then(|r| r.app.as_ref())
                        {
                            result.app = Some(app.clone());
                            break;
                        }
                    }
                }
                result
            })
            .collect()
    }

    pub fn clear(&mut self) -> bool {
        let changed = !self.records.is_empty();
        self.records.clear();
        changed
    }

    pub fn clear_transient(&mut self) -> bool {
        let before = self.records.len();
        self.records.retain(|r| {
            !matches!(
                r.state,
                ProgramStatusState::Working | ProgramStatusState::Blocked
            )
        });
        self.records.len() != before
    }

    pub fn apply(&mut self, body: &[u8]) -> bool {
        let before = self.records();
        if let Some((id, record)) = Self::parse(body) {
            if let Some(record) = record {
                self.records.retain(|r| r.id != id);
                if self.records.len() == 256 {
                    self.records.pop_front();
                }
                self.records.push_back(record);
            } else if id.is_empty() {
                self.clear();
            } else {
                self.records.retain(|r| {
                    r.id != id
                        && !r
                            .id
                            .strip_prefix(&id)
                            .is_some_and(|suffix| suffix.starts_with('/'))
                });
            }
        }
        before != self.records()
    }

    fn parse(body: &[u8]) -> Option<(String, Option<ProgramStatusRecord>)> {
        // Include OSC, selector, separator, and the longer ST terminator in the cap.
        if body.len() + 9 > 4096 {
            return None;
        }
        let mut fields = HashMap::new();
        for pair in body.split(|b| *b == b':') {
            let Ok(pair) = std::str::from_utf8(pair) else {
                continue;
            };
            let Some((key, value)) = pair.split_once('=') else {
                continue;
            };
            let (key, value) = (key.trim(), value.trim());
            if key.len() > 16 {
                return None;
            }
            if key.is_empty()
                || !key.bytes().all(|b| b.is_ascii_lowercase())
                || !value
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"_.,+/=-".contains(&b))
            {
                continue;
            }
            match key {
                "msg" => {
                    decode_text(value, 2732, 2048)?;
                }
                "title" => {
                    decode_text(value, 256, 192)?;
                }
                "id" if value.len() > 128 => return None,
                "app" if value.len() > 32 => return None,
                _ => {}
            }
            fields.insert(key, value);
        }
        let id = fields.get("id").copied().unwrap_or("");
        if fields.contains_key("id")
            && (id.is_empty() || id.split('/').count() > 8 || !id.split('/').all(valid_name))
        {
            return None;
        }
        let state = match *fields.get("state")? {
            "clear" => return Some((id.to_owned(), None)),
            "idle" => ProgramStatusState::Idle,
            "working" => ProgramStatusState::Working,
            "done" => ProgramStatusState::Done,
            "blocked" => ProgramStatusState::Blocked,
            "error" => ProgramStatusState::Error,
            _ => return None,
        };
        let kind = fields
            .get("kind")
            .filter(|&&kind| {
                state == ProgramStatusState::Blocked
                    && matches!(kind, "permission" | "question" | "auth")
            })
            .map(|s| s.to_string());
        let progress = fields
            .get("progress")
            .filter(|value| !value.is_empty() && value.bytes().all(|b| b.is_ascii_digit()))
            .and_then(|s| s.parse::<u8>().ok())
            .filter(|p| *p <= 100)
            .filter(|_| {
                matches!(
                    state,
                    ProgramStatusState::Working | ProgramStatusState::Blocked
                )
            });
        let app = fields
            .get("app")
            .filter(|s| valid_name(s))
            .map(|s| s.to_string());
        let title = fields.get("title").and_then(|s| decode_text(s, 256, 192));
        let msg = fields.get("msg").and_then(|s| decode_text(s, 2732, 2048));
        Some((
            id.to_owned(),
            Some(ProgramStatusRecord {
                id: id.to_owned(),
                state,
                kind,
                progress,
                app,
                title,
                msg,
            }),
        ))
    }
}
