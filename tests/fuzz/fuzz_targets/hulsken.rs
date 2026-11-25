#![no_main]
use isyntax::fuzz_api::Hulsken;
use libfuzzer_sys::fuzz_target;
use std::cell::RefCell;

thread_local! {
    /// Reused across inputs, as Go's FuzzDecode reuses its Scratch: a decode must not depend on what
    /// the scratch held before.
    static REUSED: RefCell<Hulsken> = RefCell::new(Hulsken::default());
}

// data = [version, planes, size][codeblock]: bit 0 of version (1 or 2), bit 0 of planes (1 or 3),
// the block side from size (4, 8, 16 or 128).
fuzz_target!(|data: &[u8]| {
    let Some((&[v, n, bw], src)) = data.split_first_chunk::<3>() else {
        return;
    };
    let (v, n, bw) = (
        1 + (v & 1),
        if n & 1 == 0 { 1 } else { 3 },
        [4, 8, 16, 128][usize::from(bw & 3)],
    );
    let reused = REUSED.with(|s| s.borrow_mut().decode(src, n, v, bw, 1 << 24));
    let fresh = Hulsken::default().decode(src, n, v, bw, 1 << 24);
    assert_eq!(reused, fresh, "the reused scratch decodes differently");
});
