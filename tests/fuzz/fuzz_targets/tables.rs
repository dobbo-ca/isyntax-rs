#![no_main]
use isyntax::fuzz_api;
use libfuzzer_sys::fuzz_target;

// data = [kind][payload]: 0 the tolerant base64 decoder, 1 the cluster table (against two fixed
// templates), 2 the XML reader of the header. Each returns an error or a value, never panics.
fuzz_target!(|data: &[u8]| {
    let Some((&kind, p)) = data.split_first() else {
        return;
    };
    match kind % 3 {
        0 => {
            let _ = fuzz_api::decode_base64(&String::from_utf8_lossy(p), 1 << 24);
        }
        1 => {
            let _ = fuzz_api::cluster_table(p, 1 << 24);
        }
        _ => {
            let _ = fuzz_api::xml(p, 1 << 24);
        }
    }
});
