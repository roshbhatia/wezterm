use super::*;
use std::assert_eq;

fn report(term: &mut TestTerm, body: &str) {
    term.print(format!("\x1b]7501;{body}\x1b\\"));
}

#[test]
fn program_status_reports_replace_and_decode() {
    let mut term = TestTerm::new(4, 20, 0);
    term.print(
        b"\x1b]7501;state=blocked:kind=permission:progress=42:app=terraform:msg=QXBwcm92ZT8\x07",
    );
    let records = term.term.get_program_status();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].state, ProgramStatusState::Blocked);
    assert_eq!(records[0].msg.as_deref(), Some("Approve?"));
    assert_eq!(records[0].progress, Some(42));
    report(&mut term, "state=done");
    let records = term.term.get_program_status();
    assert_eq!(records[0].state, ProgramStatusState::Done);
    assert_eq!(records[0].msg, None);
    assert_eq!(records[0].app, None);
    assert_eq!(records[0].progress, None);
}

#[test]
fn program_status_hierarchy_inherits_and_clears_at_segment_boundaries() {
    let mut term = TestTerm::new(4, 20, 0);
    report(&mut term, "state=idle:app=build");
    report(&mut term, "state=working:id=tests/unit");
    report(&mut term, "state=done:id=tests-other");
    assert_eq!(
        term.term.get_program_status()[1].app.as_deref(),
        Some("build")
    );
    report(&mut term, "state=idle:id=tests:app=cargo");
    let records = term.term.get_program_status();
    assert_eq!(records[1].app.as_deref(), Some("cargo"));
    report(&mut term, "state=clear:id=tests");
    assert_eq!(term.term.get_program_status().len(), 2);
    assert_eq!(term.term.get_program_status()[1].id, "tests-other");
    report(&mut term, "state=clear");
    assert!(term.term.get_program_status().is_empty());
}

#[test]
fn program_status_lifetime_tracks_prompt_exit_and_reset() {
    let mut term = TestTerm::new(4, 20, 0);
    for (id, state) in [
        ("w", "working"),
        ("b", "blocked"),
        ("d", "done"),
        ("e", "error"),
        ("i", "idle"),
    ] {
        report(&mut term, &format!("state={state}:id={id}"));
    }
    term.print(b"\x1b[?1049h\x1b[?1049l\x1b[!p");
    assert_eq!(term.term.get_program_status().len(), 5);
    term.print(b"\x1b]133;A\x07");
    assert_eq!(term.term.get_program_status().len(), 3);
    report(&mut term, "state=working:id=w");
    term.term.clear_transient_program_status();
    assert_eq!(term.term.get_program_status().len(), 3);
    term.print(b"\x1bc");
    assert!(term.term.get_program_status().is_empty());
}

#[test]
fn program_status_rejects_invalid_reports_atomically() {
    let mut term = TestTerm::new(4, 20, 0);
    report(&mut term, "state=done:app=original");
    let before = term.term.get_program_status();
    for body in [
        "state=unknown",
        "app=missing-state",
        "state=working:id=",
        "state=working:id=bad//id",
        "state=working:id=a/b/c/d/e/f/g/h/i",
        "state=working:msg=YQ=",
        "state=working:msg=Cg==",
        "state=working:title=wp8=",
        "state=working:msg=/w==",
        "state=clear:title=CQ==",
        "state=working:unknownkeytoolongxx=value",
    ] {
        report(&mut term, body);
        assert_eq!(term.term.get_program_status(), before, "{body}");
    }
    for body in [
        format!("state=working:msg={}", "YQ==".repeat(684)),
        format!("state=working:title={}", "YWFh".repeat(65)),
        format!("state=working:id={}", "a".repeat(129)),
        format!("state=working:app={}", "a".repeat(33)),
        format!("state=working:other={}", "a".repeat(4096)),
    ] {
        report(&mut term, &body);
        assert_eq!(term.term.get_program_status(), before);
    }
}

#[test]
fn program_status_ignores_malformed_pairs_and_inapplicable_values() {
    let mut term = TestTerm::new(4, 20, 0);
    report(
        &mut term,
        "state=idle:state=working:other=value:bad pair:msg=bad%:progress=+1:kind=auth:app=bad/name",
    );
    let records = term.term.get_program_status();
    assert_eq!(records[0].state, ProgramStatusState::Working);
    assert_eq!(records[0].msg, None);
    assert_eq!(records[0].progress, None);
    assert_eq!(records[0].kind, None);
    assert_eq!(records[0].app, None);
    report(&mut term, " state = blocked : progress = 100 :kind=unknown");
    assert_eq!(term.term.get_program_status()[0].progress, Some(100));
    assert_eq!(term.term.get_program_status()[0].kind, None);
}

#[test]
fn program_status_evicts_least_recently_updated_record() {
    let mut term = TestTerm::new(4, 20, 0);
    for i in 0..256 {
        report(&mut term, &format!("state=done:id=task{i}"));
    }
    report(&mut term, "state=error:id=task0");
    report(&mut term, "state=done:id=task256");
    let records = term.term.get_program_status();
    assert_eq!(records.len(), 256);
    assert!(records.iter().any(|r| r.id == "task0"));
    assert!(!records.iter().any(|r| r.id == "task1"));
}

#[derive(Clone)]
struct ReplyWriter(Arc<Mutex<Vec<u8>>>);
impl std::io::Write for ReplyWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn program_status_feature_detection_replies_without_mutating_records() {
    let bytes = Arc::new(Mutex::new(Vec::new()));
    let mut term = Terminal::new(
        TerminalSize::default(),
        Arc::new(TestTermConfig { scrollback: 0 }),
        "WezTerm",
        "test",
        Box::new(ReplyWriter(bytes.clone())),
    );
    term.advance_bytes(b"\x1b]7501;?\x1b\\");
    for _ in 0..100 {
        if !bytes.lock().unwrap().is_empty() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert_eq!(*bytes.lock().unwrap(), b"\x1b]7501;?\x1b\\");
    assert!(term.get_program_status().is_empty());
}
