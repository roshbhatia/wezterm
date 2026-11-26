use super::TestTerm;
use sha2::{Digest, Sha256};
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn program_resume_reports_only_reach_the_pane_that_issued_the_token() {
    let token = "independent-test-pane-token";
    let time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let body = format!("pid=74877:time={time}:cwd=L3RtcA==:argv=L2Jpbi9lY2hv,");
    let digest = hex::encode(Sha256::digest(format!("{body}:{token}")));
    let report = format!("\x1b]7577;{body}:mac={digest}\x1b\\");

    let mut source = TestTerm::new(4, 20, 0);
    let mut other = TestTerm::new(4, 20, 0);
    source.term.set_program_resume_token(token.to_owned());
    other
        .term
        .set_program_resume_token("a-different-test-pane-token".to_owned());
    source.print(&report);
    other.print(&report);

    let record = source.term.get_program_resume(74877).unwrap();
    assert_eq!(record.argv, ["/bin/echo", ""]);
    assert_eq!(record.cwd.as_deref(), Some("/tmp"));
    assert!(other.term.get_program_resume(74877).is_none());
}
