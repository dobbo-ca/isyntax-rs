//! The XML header, read the way Go's `encoding/xml` reads it into the `xObject` / `xAttr` structs
//! of `internal/container`: a root element whose `Attribute` children carry `Group`, `Element`,
//! their text, and the `DataObject`s of an `Array` child. Anything else is skipped. Strict mode:
//! a syntax error anywhere before the end of the root is an error, as in Go. Not a general XML
//! parser: no DTD, no namespaces (a prefix is ignored), UTF-8 only.

use crate::Error;
use crate::budget::Acct;
use crate::xml_names::is_name;

/// Filled elements deeper than this are an error, as in Go; skipped subtrees nest freely (the
/// `Acct` bounds them).
const MAX_DEPTH: usize = 10000;
/// What an element is charged for, besides its text.
const NODE_COST: u64 = 160;

#[derive(Default, Debug)]
pub(crate) struct XObject {
    pub attrs: Vec<XAttr>,
}

#[derive(Default, Debug)]
pub(crate) struct XAttr {
    /// `Group` and `Element` as numbers; `None` unless both parse (`strconv.ParseUint(s, 0, 16)`).
    pub id: Option<(u16, u16)>,
    pub value: String,
    pub objects: Vec<XObject>,
}

impl XObject {
    /// The first direct attribute with this tag.
    pub fn attr(&self, group: u16, elem: u16) -> Option<&XAttr> {
        self.attrs.iter().find(|a| a.id == Some((group, elem)))
    }

    /// The first attribute with this tag at any depth.
    pub fn find(&self, group: u16, elem: u16) -> Option<&XAttr> {
        for a in &self.attrs {
            if a.id == Some((group, elem)) {
                return Some(a);
            }
            for o in &a.objects {
                if let Some(f) = o.find(group, elem) {
                    return Some(f);
                }
            }
        }
        None
    }

    /// The `DataObject`s of the array attribute with this tag.
    pub fn objects(&self, group: u16, elem: u16) -> &[XObject] {
        match self.attr(group, elem) {
            Some(a) => &a.objects,
            None => &[],
        }
    }
}

enum Frame {
    Obj(XObject),
    Attr(XAttr),
    Array(Vec<XObject>),
    Skip,
}

fn syn(msg: &str) -> Error {
    Error::Xml(msg.to_string())
}

/// Parses the first element of `b` into an object, charging `acct` for what it keeps.
pub(crate) fn parse(b: &[u8], acct: &mut Acct) -> Result<XObject, Error> {
    let mut p = Parser { b, i: 0 };
    let mut stack: Vec<(Frame, &[u8])> = Vec::new();
    loop {
        acct.tick()?;
        let Some(&c) = b.get(p.i) else {
            return Err(syn(if stack.is_empty() {
                "EOF"
            } else {
                "unexpected EOF"
            }));
        };
        if c != b'<' {
            // Text is built, and charged first, only where it is kept.
            let keep = matches!(stack.last(), Some((Frame::Attr(_), _)));
            let text = p.text(None, false, keep.then_some(&mut *acct))?;
            if let Some((Frame::Attr(a), _)) = stack.last_mut() {
                append(&mut a.value, text, acct)?;
            }
            continue;
        }
        match b.get(p.i + 1) {
            Some(b'/') => {
                p.i += 2;
                let name = p.name("expected element name after </")?;
                p.space();
                if b.get(p.i) != Some(&b'>') {
                    return Err(syn("invalid characters between </name and >"));
                }
                p.i += 1;
                match stack.last() {
                    Some((_, open)) if *open == name.as_bytes() => {}
                    _ => return Err(syn("element closing tag mismatch")),
                }
                if let Some(root) = pop(&mut stack) {
                    return Ok(root);
                }
            }
            Some(b'?') => p.proc_inst()?,
            Some(b'!') => {
                let keep = matches!(stack.last(), Some((Frame::Attr(_), _)));
                let text = p.bang(keep.then_some(&mut *acct))?;
                if let (Some(text), Some((Frame::Attr(a), _))) = (text, stack.last_mut()) {
                    append(&mut a.value, text, acct)?;
                }
            }
            _ => {
                let (name, attrs, empty) = p.start_tag(acct)?;
                acct.add(NODE_COST)?;
                let local = local_name(name);
                let frame = match stack.last() {
                    None => Frame::Obj(XObject::default()),
                    Some((Frame::Obj(_), _)) if local == "Attribute" => Frame::Attr(XAttr {
                        id: tag(&attrs),
                        ..XAttr::default()
                    }),
                    Some((Frame::Attr(_), _)) if local == "Array" => Frame::Array(Vec::new()),
                    Some((Frame::Array(_), _)) if local == "DataObject" => {
                        Frame::Obj(XObject::default())
                    }
                    _ => Frame::Skip,
                };
                if matches!(frame, Frame::Obj(_) | Frame::Attr(_)) && stack.len() >= MAX_DEPTH {
                    return Err(syn("exceeded max depth"));
                }
                stack.push((frame, name.as_bytes()));
                if empty && let Some(root) = pop(&mut stack) {
                    return Ok(root);
                }
            }
        }
    }
}

/// Appends `s`, already charged, to `v`, growing `v` by doubling and charging the growth, so the
/// capacity held is the capacity charged.
fn append(v: &mut String, s: String, acct: &mut Acct) -> Result<(), Error> {
    if v.is_empty() {
        *v = s;
        return Ok(());
    }
    let want = v.len().saturating_add(s.len());
    if want > v.capacity() {
        let grow = want.max(v.capacity().saturating_mul(2));
        acct.add((grow - v.capacity()) as u64)?;
        v.try_reserve_exact(grow - v.len())
            .map_err(|_| Error::OutOfMemory)?;
    }
    v.push_str(&s);
    Ok(())
}

/// Closes the innermost element; returns the root object when that was the root.
fn pop(stack: &mut Vec<(Frame, &[u8])>) -> Option<XObject> {
    let (f, _) = stack.pop()?;
    match (f, stack.last_mut()) {
        (Frame::Obj(o), Some((Frame::Array(v), _))) => v.push(o),
        (Frame::Obj(o), None) => return Some(o),
        (Frame::Array(v), Some((Frame::Attr(a), _))) => a.objects.extend(v),
        (Frame::Attr(a), Some((Frame::Obj(o), _))) => o.attrs.push(a),
        _ => {}
    }
    None
}

/// Go's `Name.Local`: the part after a single colon.
fn local_name(s: &str) -> &str {
    if s.matches(':').count() > 1 {
        return s;
    }
    match s.split_once(':') {
        Some((space, local)) if !space.is_empty() && !local.is_empty() => local,
        _ => s,
    }
}

/// The `Group` and `Element` attributes (the last of each wins, as in Go).
fn tag(attrs: &[(&str, String)]) -> Option<(u16, u16)> {
    let get = |name: &str| {
        attrs
            .iter()
            .rev()
            .find(|(k, _)| local_name(k) == name)
            .and_then(|(_, v)| parse_uint16(v))
    };
    Some((get("Group")?, get("Element")?))
}

/// `strconv.ParseUint(s, 0, 16)`: a `0x`, `0b`, `0o` or leading-0 prefix, underscores between
/// digits; `None` for any error, including a value over 65535.
pub(crate) fn parse_uint16(s: &str) -> Option<u16> {
    let b = s.as_bytes();
    let first = *b.first()?;
    let (base, digits) = if first == b'0' {
        match (b.len() >= 3, b.get(1).map(u8::to_ascii_lowercase)) {
            (true, Some(b'b')) => (2, &b[2..]),
            (true, Some(b'o')) => (8, &b[2..]),
            (true, Some(b'x')) => (16, &b[2..]),
            _ => (8, &b[1..]),
        }
    } else {
        (10, b)
    };
    let mut n: u64 = 0;
    let mut underscores = false;
    for &c in digits {
        let d = match c {
            b'_' => {
                underscores = true;
                continue;
            }
            b'0'..=b'9' => c - b'0',
            b'a'..=b'z' => c - b'a' + 10,
            b'A'..=b'Z' => c - b'A' + 10,
            _ => return None,
        };
        if u64::from(d) >= base {
            return None;
        }
        n = n * base + u64::from(d);
        if n > u64::from(u16::MAX) {
            return None;
        }
    }
    if underscores && !underscore_ok(b) {
        return None;
    }
    u16::try_from(n).ok()
}

/// `strconv.underscoreOK`.
fn underscore_ok(s: &[u8]) -> bool {
    let mut saw = b'^';
    let mut i = 0;
    let mut hex = false;
    if s.len() >= 2 && s[0] == b'0' && matches!(s[1].to_ascii_lowercase(), b'b' | b'o' | b'x') {
        i = 2;
        saw = b'0';
        hex = s[1].eq_ignore_ascii_case(&b'x');
    }
    while i < s.len() {
        let c = s[i];
        i += 1;
        if c.is_ascii_digit() || (hex && matches!(c.to_ascii_lowercase(), b'a'..=b'f')) {
            saw = b'0';
            continue;
        }
        if c == b'_' {
            if saw != b'0' {
                return false;
            }
            saw = b'_';
            continue;
        }
        if saw == b'_' {
            return false;
        }
        saw = b'!';
    }
    saw != b'_'
}

fn in_range(c: char) -> bool {
    matches!(c as u32, 0x09 | 0x0A | 0x0D | 0x20..=0xD7FF | 0xE000..=0xFFFD | 0x10000..=0x10FFFF)
}

fn is_name_byte(c: u8) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, b'_' | b':' | b'.' | b'-') || c >= 0x80
}

/// Where the bytes of `text` go: counted and checked as UTF-8 and XML characters, and stored once
/// `out` is set (after the count has been charged). The errors are Go's, and come in its order
/// (invalid UTF-8 first), after the whole text has been read.
#[derive(Default)]
struct Sink {
    out: Option<Vec<u8>>,
    n: usize,
    pend: [u8; 4],
    plen: usize,
    bad_utf8: bool,
    bad_char: Option<char>,
}

impl Sink {
    fn push(&mut self, c: u8) {
        if let Some(o) = &mut self.out {
            o.push(c);
            return;
        }
        self.n += 1;
        if self.plen == 0 && c < 0x80 {
            self.check(char::from(c));
            return;
        }
        if let Some(slot) = self.pend.get_mut(self.plen) {
            *slot = c;
        }
        self.plen += 1;
        match std::str::from_utf8(&self.pend[..self.plen]) {
            Ok(s) => {
                let ch = s.chars().next();
                self.plen = 0;
                if let Some(ch) = ch {
                    self.check(ch);
                }
            }
            Err(e) if e.error_len().is_some() => {
                self.bad_utf8 = true;
                self.plen = 0;
            }
            Err(_) => {}
        }
    }

    fn check(&mut self, c: char) {
        if !in_range(c) && self.bad_char.is_none() {
            self.bad_char = Some(c);
        }
    }

    /// Drops the `]]` that ended a CDATA section.
    fn cut_two(&mut self) {
        match &mut self.out {
            Some(o) => o.truncate(o.len().saturating_sub(2)),
            None => self.n = self.n.saturating_sub(2),
        }
    }

    fn finish(&self) -> Result<(), Error> {
        if self.bad_utf8 || self.plen > 0 {
            return Err(syn("invalid UTF-8"));
        }
        match self.bad_char {
            Some(bad) => Err(Error::Xml(format!(
                "illegal character code U+{:04X}",
                bad as u32
            ))),
            None => Ok(()),
        }
    }
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn space(&mut self) {
        while matches!(self.b.get(self.i), Some(b' ' | b'\r' | b'\n' | b'\t')) {
            self.i += 1;
        }
    }

    fn name(&mut self, what: &str) -> Result<&'a str, Error> {
        let start = self.i;
        match self.b.get(start) {
            Some(&c) if c.is_ascii_alphabetic() || c == b'_' || c == b':' || c >= 0x80 => {}
            _ => return Err(syn(what)),
        }
        while self.b.get(self.i).is_some_and(|&c| is_name_byte(c)) {
            self.i += 1;
        }
        let b = self.b;
        let name = std::str::from_utf8(&b[start..self.i]).map_err(|_| syn("invalid XML name"))?;
        if !is_name(name) {
            return Err(Error::Xml(format!("invalid XML name: {name}")));
        }
        Ok(name)
    }

    /// Character data up to a `<` (left unread), or up to the closing `quote` or `]]>` (read).
    /// Entities are decoded, `\r\n` and `\r` become `\n`, and the result is checked for UTF-8 and
    /// characters XML allows. With `keep`, the text is charged to it and returned; without, it is
    /// only checked and the result is empty.
    fn text(
        &mut self,
        quote: Option<u8>,
        cdata: bool,
        keep: Option<&mut Acct>,
    ) -> Result<String, Error> {
        let start = self.i;
        let mut sink = Sink::default();
        self.scan(quote, cdata, &mut sink)?;
        sink.finish()?;
        let Some(acct) = keep else {
            return Ok(String::new());
        };
        acct.add(sink.n as u64)?;
        // The `]]` of a CDATA end is pushed, then cut.
        let mut out = Vec::new();
        out.try_reserve_exact(sink.n + 2)
            .map_err(|_| Error::OutOfMemory)?;
        sink.out = Some(out);
        self.i = start;
        self.scan(quote, cdata, &mut sink)?;
        String::from_utf8(sink.out.unwrap_or_default()).map_err(|_| syn("invalid UTF-8"))
    }

    /// The loop of `text`, feeding the decoded bytes to `sink`.
    fn scan(&mut self, quote: Option<u8>, cdata: bool, sink: &mut Sink) -> Result<(), Error> {
        let (mut b0, mut b1) = (0u8, 0u8);
        loop {
            let Some(&c) = self.b.get(self.i) else {
                if cdata || quote.is_some() {
                    return Err(syn("unexpected EOF"));
                }
                return Ok(());
            };
            if quote.is_none() && b0 == b']' && b1 == b']' && c == b'>' {
                self.i += 1;
                if cdata {
                    sink.cut_two();
                    return Ok(());
                }
                return Err(syn("unescaped ]]> not in CDATA section"));
            }
            if c == b'<' && !cdata {
                if quote.is_some() {
                    return Err(syn("unescaped < inside quoted string"));
                }
                return Ok(());
            }
            if quote == Some(c) {
                self.i += 1;
                return Ok(());
            }
            self.i += 1;
            if c == b'&' && !cdata {
                for e in self.entity()? {
                    sink.push(e);
                }
                (b0, b1) = (0, 0);
                continue;
            }
            if c == b'\r' {
                sink.push(b'\n');
            } else if !(b1 == b'\r' && c == b'\n') {
                sink.push(c);
            }
            (b0, b1) = (b1, c);
        }
    }

    /// After a `&`: the UTF-8 of the character it names.
    fn entity(&mut self) -> Result<Vec<u8>, Error> {
        let bad = || syn("invalid character entity");
        if self.b.get(self.i) == Some(&b'#') {
            self.i += 1;
            let mut base = 10;
            if self.b.get(self.i) == Some(&b'x') {
                base = 16;
                self.i += 1;
            }
            let start = self.i;
            while self
                .b
                .get(self.i)
                .is_some_and(|c| c.is_ascii_digit() || (base == 16 && c.is_ascii_hexdigit()))
            {
                self.i += 1;
            }
            if self.b.get(self.i) != Some(&b';') {
                return Err(bad());
            }
            let digits = std::str::from_utf8(&self.b[start..self.i]).map_err(|_| bad())?;
            self.i += 1;
            let n = u64::from_str_radix(digits, base).map_err(|_| bad())?;
            if n > 0x10FFFF {
                return Err(bad());
            }
            let ch = char::from_u32(n as u32).unwrap_or('\u{FFFD}');
            return Ok(ch.to_string().into_bytes());
        }
        let start = self.i;
        while self.b.get(self.i).is_some_and(|&c| is_name_byte(c)) {
            self.i += 1;
        }
        let name = &self.b[start..self.i];
        if self.b.get(self.i) != Some(&b';') {
            return Err(bad());
        }
        self.i += 1;
        let r: &[u8] = match name {
            b"lt" => b"<",
            b"gt" => b">",
            b"amp" => b"&",
            b"apos" => b"'",
            b"quot" => b"\"",
            _ => return Err(bad()),
        };
        Ok(r.to_vec())
    }

    /// `<name attr="v" ...>` or `<name ... />`: the name, the attributes, whether it is empty.
    #[allow(clippy::type_complexity)]
    fn start_tag(
        &mut self,
        acct: &mut Acct,
    ) -> Result<(&'a str, Vec<(&'a str, String)>, bool), Error> {
        self.i += 1;
        let name = self.name("expected element name after <")?;
        let mut attrs = Vec::new();
        loop {
            self.space();
            match self.b.get(self.i) {
                None => return Err(syn("unexpected EOF")),
                Some(b'/') => {
                    self.i += 1;
                    if self.b.get(self.i) != Some(&b'>') {
                        return Err(syn("expected /> in element"));
                    }
                    self.i += 1;
                    return Ok((name, attrs, true));
                }
                Some(b'>') => {
                    self.i += 1;
                    return Ok((name, attrs, false));
                }
                Some(_) => {}
            }
            let key = self.name("expected attribute name in element")?;
            self.space();
            if self.b.get(self.i) != Some(&b'=') {
                return Err(syn("attribute name without = in element"));
            }
            self.i += 1;
            self.space();
            let q = match self.b.get(self.i) {
                Some(&q) if q == b'"' || q == b'\'' => q,
                _ => return Err(syn("unquoted or missing attribute value in element")),
            };
            self.i += 1;
            // Only `Group` and `Element` are read, the last of each: nothing else is built, so
            // the attributes of a tag cost no memory however many there are.
            let l = local_name(key);
            let keep = l == "Group" || l == "Element";
            let v = self.text(Some(q), false, keep.then_some(&mut *acct))?;
            if keep {
                match attrs.iter_mut().find(|(k, _)| local_name(k) == l) {
                    Some(e) => *e = (key, v),
                    None => attrs.push((key, v)),
                }
            }
        }
    }

    /// `<?target data?>`: checks the version and encoding of `<?xml ...?>`.
    fn proc_inst(&mut self) -> Result<(), Error> {
        self.i += 2;
        let target = self.name("expected target name after <?")?;
        self.space();
        let rest = &self.b[self.i..];
        let Some(end) = rest.windows(2).position(|w| w == b"?>") else {
            return Err(syn("unexpected EOF"));
        };
        let content = String::from_utf8_lossy(&rest[..end]).into_owned();
        self.i += end + 2;
        if target == "xml" {
            let ver = proc_param("version", &content);
            if !ver.is_empty() && ver != "1.0" {
                return Err(Error::Xml(format!("unsupported version: {ver}")));
            }
            let enc = proc_param("encoding", &content);
            if !enc.is_empty() && !enc.eq_ignore_ascii_case("utf-8") {
                return Err(Error::Xml(format!("encoding {enc:?} declared")));
            }
        }
        Ok(())
    }

    /// `<!...`: a comment (nothing), a CDATA section (its text) or a directive (nothing).
    fn bang(&mut self, keep: Option<&mut Acct>) -> Result<Option<String>, Error> {
        let rest = &self.b[self.i + 2..];
        if rest.starts_with(b"-") {
            if !rest.starts_with(b"--") {
                return Err(syn("invalid sequence <!- not part of <!--"));
            }
            let body = &rest[2..];
            let Some(at) = body.windows(2).position(|w| w == b"--") else {
                return Err(syn("unexpected EOF"));
            };
            if body.get(at + 2) != Some(&b'>') {
                return Err(syn("invalid sequence \"--\" not allowed in comments"));
            }
            self.i += 2 + 2 + at + 3;
            return Ok(None);
        }
        if rest.starts_with(b"[") {
            if !rest.starts_with(b"[CDATA[") {
                return Err(syn("invalid <![ sequence"));
            }
            self.i += 2 + 7;
            return self.text(None, true, keep).map(Some);
        }
        // A directive: up to the matching `>`, with quotes, nested `<>` and `<!-- -->`. The first
        // byte after `<!` is not looked at (Go writes it to the buffer unexamined).
        if rest.is_empty() {
            return Err(syn("unexpected EOF"));
        }
        let eof = || syn("unexpected EOF");
        let mut quote = 0u8;
        let mut depth = 0usize;
        let mut j = self.i + 3;
        loop {
            let mut c = *self.b.get(j).ok_or_else(eof)?;
            j += 1;
            if quote == 0 && c == b'>' && depth == 0 {
                break;
            }
            loop {
                if c == quote {
                    quote = 0;
                } else if quote == 0 {
                    match c {
                        b'\'' | b'"' => quote = c,
                        b'>' => depth = depth.saturating_sub(1),
                        b'<' => {
                            let mut comment = true;
                            for &want in b"!--" {
                                let d = *self.b.get(j).ok_or_else(eof)?;
                                j += 1;
                                if d != want {
                                    depth += 1;
                                    c = d;
                                    comment = false;
                                    break;
                                }
                            }
                            if !comment {
                                continue;
                            }
                            let at = self.b[j..].windows(3).position(|w| w == b"-->");
                            j += at.ok_or_else(eof)? + 3;
                        }
                        _ => {}
                    }
                }
                break;
            }
        }
        self.i = j;
        Ok(None)
    }
}

/// Go's `procInst`: the quoted value of `param=` in a processing instruction. Like Go it moves on
/// to the next `param=` until one is followed by a quote.
fn proc_param<'s>(param: &str, s: &'s str) -> &'s str {
    let key = format!("{param}=");
    let sb = s.as_bytes();
    let mut i = 0;
    let mut sep = 0u8;
    while i < sb.len() {
        let sub = &sb[i..];
        let Some(k) = sub.windows(key.len()).position(|w| w == key.as_bytes()) else {
            return "";
        };
        let Some(&c) = sub.get(key.len() + k) else {
            return "";
        };
        i += key.len() + k + 1;
        if c == b'\'' || c == b'"' {
            sep = c;
            break;
        }
    }
    if sep == 0 {
        return "";
    }
    let Some(j) = sb[i..].iter().position(|&c| c == sep) else {
        return "";
    };
    s.get(i..i + j).unwrap_or("")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_str(s: &str) -> Result<XObject, Error> {
        parse(s.as_bytes(), &mut Acct::new(1 << 30))
    }

    #[test]
    fn tag_attributes_other_than_group_and_element_are_dropped() {
        let many = " b=\"\"".repeat(10_000);
        let x = parse_str(&format!(
            "<DataObject{many}><Attribute x=\"\"{many} Group=\"0x1\" Group=\"0x2\" Element=\"0x3\"/></DataObject>"
        ))
        .unwrap();
        assert!(x.attr(2, 3).is_some());
    }

    #[test]
    fn objects_and_text() {
        let x = parse_str(
            "<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n<DataObject>\
             <Attribute Group=\"0x301D\" Element=\"0x1003\"><Array>\
             <DataObject><Attribute Group='0x0008' Element='0x002A'> a &amp; b&#x0D;&#65;<![CDATA[<c>]]> </Attribute></DataObject>\
             </Array></Attribute><Other><Attribute Group=\"0x1\" Element=\"0x2\"/></Other></DataObject>",
        )
        .unwrap();
        assert_eq!(x.attrs.len(), 1);
        let a = x.attr(0x301D, 0x1003).unwrap();
        assert_eq!(a.objects.len(), 1);
        let t = x.find(0x0008, 0x002A).unwrap();
        assert_eq!(t.value, " a & b\rA<c> ");
        assert!(x.find(0x1, 0x2).is_none(), "Other is skipped");
    }

    /// The least `max_alloc` that parses `s`.
    fn need(s: &str) -> u64 {
        let (mut lo, mut hi) = (0u64, 1 << 20);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if parse(s.as_bytes(), &mut Acct::new(mid)).is_ok() {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        lo
    }

    #[test]
    fn text_is_charged_where_it_is_kept() {
        let big = "a".repeat(5000);
        let attr = |body: &str| {
            format!("<r><Attribute Group=\"0x1\" Element=\"0x2\">{body}</Attribute></r>")
        };
        // Two elements, the 6 bytes of Group and Element, and the text of the one that keeps it.
        for (s, want) in [
            (attr(&big), 2 * NODE_COST + 6 + 5000),
            (
                attr(&format!("<![CDATA[{big}]]>")),
                2 * NODE_COST + 6 + 5000,
            ),
            (
                format!(
                    "<r><Attribute Group=\"0x1{}\" Element=\"0x2\"/></r>",
                    " ".repeat(5000)
                ),
                2 * NODE_COST + 6 + 5000,
            ),
            // Text of a frame that drops it costs nothing.
            (format!("<r><Other>{big}</Other></r>"), 2 * NODE_COST),
            (format!("<r>{big}<!--x--></r>"), NODE_COST),
            (format!("<r><Other x=\"{big}\"/></r>"), 2 * NODE_COST),
        ] {
            let n = need(&s);
            assert_eq!(n, want, "{}", &s[..40]);
            assert!(parse(s.as_bytes(), &mut Acct::new(n)).is_ok());
            if n > 0 {
                assert_eq!(
                    parse(s.as_bytes(), &mut Acct::new(n - 1)).err(),
                    Some(Error::AllocLimit)
                );
            }
        }
    }

    #[test]
    fn syntax_errors_come_before_the_limit() {
        let s = "<a><Attribute Group=\"0x1\" Element=\"0x2\">xxxx&bogus;</Attribute></a>";
        assert!(matches!(
            parse(s.as_bytes(), &mut Acct::new(2 * NODE_COST + 6)),
            Err(Error::Xml(_))
        ));
        let s = "<a><Attribute Group=\"0x1\" Element=\"0x2\">xxxx\u{1}</Attribute></a>";
        assert!(matches!(
            parse(s.as_bytes(), &mut Acct::new(2 * NODE_COST + 6)),
            Err(Error::Xml(_))
        ));
    }

    #[test]
    fn pieces_of_one_value_join() {
        let x = parse_str(
            "<r><Attribute Group=\"0x1\" Element=\"0x2\">x<!---->\u{e9}<![CDATA[z]]>&#65;\r\nw</Attribute></r>",
        )
        .unwrap();
        assert_eq!(x.attr(1, 2).unwrap().value, "x\u{e9}zA\nw");
    }

    #[test]
    fn errors() {
        for s in [
            "",
            "hello",
            "<a>",
            "<a></b>",
            "<a b=1></a>",
            "<a b></a>",
            "<a>&bogus;</a>",
            "<a>&#0;</a>",
            "<a>x ]]> y</a>",
            "<a b=\"<\"></a>",
            "<?xml version=\"2.0\"?><a/>",
            "<?xml version=\"1.0\" encoding=\"UTF-16\"?><a/>",
            "<a><!-- x -- y --></a>",
            "<a>\u{1}</a>",
            "<a\u{D7}/>",
            "<a b\u{D7}=\"1\"/>",
            "<a>x</a\u{D7}>",
            "<a><!-x></a>",
            "<a><!-></a>",
            "<?xml version=1 version=\"2.0\"?><a/>",
            "<?xml encoding=x encoding=\"UTF-16\"?><a/>",
        ] {
            assert!(parse_str(s).is_err(), "{s:?}");
        }
        assert!(parse(b"<a>\xff</a>", &mut Acct::new(1 << 30)).is_err());
        assert!(parse_str("<a/>trailing garbage").is_ok());
        assert!(parse_str("<a><b/></a>").is_ok());
        for ok in [
            "<\u{E9}\u{B7} \u{4E00}-1=\"x\"/>",
            "<a><!>x></a>",
            "<a><!'>x'></a>",
            "<a><!<>x></a>",
            "<a><!DOCTYPE x <!-- > --> \"y>\" <z>>x></a>",
            "<?xml version=\"1.0\" version=\"2.0\"?><a/>",
        ] {
            assert!(parse_str(ok).is_ok(), "{ok:?}");
        }
    }

    #[test]
    fn proc_params() {
        assert_eq!(proc_param("version", "version=1 version=\"2.0\""), "2.0");
        assert_eq!(proc_param("version", "version='1.0'"), "1.0");
        assert_eq!(proc_param("version", "version="), "");
        assert_eq!(proc_param("version", "version=\"x"), "");
        assert_eq!(proc_param("version", "version=\u{E9}\"a\""), "");
    }

    #[test]
    fn nesting_limit() {
        // Filled elements (3 per Attribute/Array/DataObject round) are limited, skipped ones are not.
        let deep = "<a>".to_string() + &"<Attribute><Array><DataObject>".repeat(MAX_DEPTH / 3 + 1);
        assert!(matches!(parse_str(&deep), Err(Error::Xml(m)) if m.contains("max depth")));
        let n = 3 * MAX_DEPTH;
        let skipped = format!("<a>{}{}</a>", "<x>".repeat(n), "</x>".repeat(n));
        assert!(parse_str(&skipped).is_ok());
    }

    #[test]
    fn node_charge() {
        let x = "<a>".to_string() + &"<b/>".repeat(100) + "</a>";
        assert!(parse(x.as_bytes(), &mut Acct::new(NODE_COST * 10)).is_err());
    }

    #[test]
    fn cancel_between_elements() {
        let c = std::sync::atomic::AtomicBool::new(true);
        let x = "<a>".to_string() + &"<b/>".repeat(5000) + "</a>";
        let r = parse(x.as_bytes(), &mut Acct::with_cancel(u64::MAX, &c));
        assert_eq!(r.err(), Some(Error::Cancelled));
    }

    #[test]
    fn uint16() {
        for (s, want) in [
            ("0x301D", Some(0x301D)),
            ("0X301d", Some(0x301D)),
            ("12", Some(12)),
            ("0", Some(0)),
            ("010", Some(8)),
            ("0b101", Some(5)),
            ("0o17", Some(15)),
            ("1_000", Some(1000)),
            ("0x_1F", Some(0x1F)),
            ("0x10000", None),
            ("65536", None),
            ("1__0", None),
            ("_1", None),
            ("1_", None),
            ("", None),
            ("0x", None),
            ("-1", None),
            ("+1", None),
            ("0xg", None),
        ] {
            assert_eq!(parse_uint16(s), want, "{s:?}");
        }
    }
}
