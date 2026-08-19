//! Strip markdown syntax so it isn't read aloud.
//!
//! Listening to a `.md` file through espeak-ng is noisy: every `*` in a nested
//! `** bold **` bullet is pronounced, `#` announces itself, and a fenced code
//! block or a table turns into a long run of punctuation names. This module runs
//! over the *whole document* before [`crate::kokoro::split_sentences`] chunks it
//! (the fenced-code state machine is inherently multi-line, so it cannot run
//! per-chunk) and removes the syntax while keeping the prose.
//!
//! The rules are deliberately conservative — a false positive silently mangles
//! speech, which is worse than leaving a stray marker in. In particular
//! `3 * 4 = 12` and `snake_case_name` pass through untouched; see [`strip`].
//!
//! Not handled on purpose: 4-space indented code blocks. They are
//! indistinguishable from a wrapped list continuation, so dropping them would
//! silently eat prose.

/// Strip markdown syntax from `text`, preserving line structure.
///
/// Line breaks survive because `split_sentences` treats `\n` as a chunk
/// boundary — one list item or heading per utterance is exactly the phrasing we
/// want. Lines that become empty are dropped.
pub fn strip(text: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    // Emphasis is resolved per *paragraph*, not per line: markdown authors wrap
    // `**bold phrases**` across a line break all the time, and a line-local pass
    // leaves the orphaned markers to be read aloud. A paragraph is also the right
    // upper bound — pairing across a whole document could match a stray `*` on
    // line 5 with an unrelated one 400 lines later.
    let mut para: Vec<String> = Vec::new();
    let mut fence: Option<char> = None;

    // Anything dropped wholesale (blank line, fence, table, rule) also ends the
    // paragraph, so emphasis never pairs across a structural break.
    macro_rules! end_para {
        () => {
            if !para.is_empty() {
                out.extend(finish_paragraph(&para));
                para.clear();
            }
        };
    }

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            end_para!();
            continue;
        }

        // Fenced code: ``` or ~~~ (3+). Only the same marker closes the fence, so
        // a ``` inside a ~~~ block doesn't end it early.
        if let Some(marker) = fence_marker(trimmed) {
            match fence {
                Some(open) if open == marker => fence = None,
                Some(_) => {}
                None => fence = Some(marker),
            }
            end_para!();
            continue;
        }
        // An unterminated fence swallows the rest of the input, which is the same
        // thing every markdown renderer does.
        if fence.is_some() {
            continue;
        }

        // Tables: any row, plus the |---|---| separator. Read linearly they make
        // no sense, so the whole block goes.
        if trimmed.starts_with('|') {
            end_para!();
            continue;
        }
        // Horizontal rules (---, ***, ___) and setext underlines (===).
        if is_rule(trimmed) {
            end_para!();
            continue;
        }
        // Link definitions ([ref]: http://…) carry no prose at all.
        if is_link_definition(trimmed) {
            end_para!();
            continue;
        }

        let line = strip_block_prefixes(line);
        para.push(strip_inline(&line));
    }
    end_para!();

    out.join("\n")
}

/// Finish one paragraph: resolve emphasis and escapes across its lines, then hand
/// back the non-empty ones. Line structure survives because `split_sentences`
/// treats `\n` as a chunk boundary.
fn finish_paragraph(lines: &[String]) -> Vec<String> {
    let joined = lines.join("\n");
    let joined = strip_emphasis(&joined);
    let joined = strip_escapes(&joined);
    joined
        .lines()
        // Removing a marker leaves a hole (`text ![img](u) more`); collapse the
        // runs so the spoken text — and --show-text — reads cleanly.
        .map(|l| l.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|l| !l.is_empty())
        .collect()
}

/// The fence character if `trimmed` opens or closes a code fence, else `None`.
fn fence_marker(trimmed: &str) -> Option<char> {
    for marker in ['`', '~'] {
        let run = trimmed.chars().take_while(|&c| c == marker).count();
        if run >= 3 {
            return Some(marker);
        }
    }
    None
}

/// `---`, `***`, `___` (3+ of one marker, spaces allowed) or a setext `===` rule.
fn is_rule(trimmed: &str) -> bool {
    for marker in ['-', '*', '_', '='] {
        let count = trimmed.chars().filter(|&c| c == marker).count();
        if count >= 3 && trimmed.chars().all(|c| c == marker || c == ' ') {
            return true;
        }
    }
    false
}

/// `[ref]: https://…` — a definition line, never spoken prose. A *footnote*
/// definition (`[^1]: …`) looks the same but carries real text, so it is excluded
/// here and loses only its label in [`strip_block_prefixes`].
fn is_link_definition(trimmed: &str) -> bool {
    let Some(rest) = trimmed.strip_prefix('[') else { return false };
    if rest.starts_with('^') {
        return false;
    }
    let Some(close) = rest.find(']') else { return false };
    rest[close + 1..].starts_with(':')
}

/// Remove leading block markers: blockquote `>`, heading `#`, list bullets, task
/// checkboxes, footnote-definition labels. Ordered-list markers (`1.`) are kept —
/// the number is meaningful when you're listening rather than looking.
fn strip_block_prefixes(line: &str) -> String {
    let mut s = line.trim_start();

    // Blockquotes nest (`> > quoted`), so peel repeatedly.
    while let Some(rest) = s.strip_prefix('>') {
        s = rest.trim_start();
    }

    // ATX heading: 1-6 hashes then whitespace (or a bare `#` line).
    let hashes = s.chars().take_while(|&c| c == '#').count();
    let is_heading = (1..=6).contains(&hashes)
        && s[hashes..].chars().next().is_none_or(char::is_whitespace);
    if is_heading {
        // Closing hashes (`## Title ##`) are decoration, not text.
        let body = s[hashes..].trim().trim_end_matches('#').trim();
        if body.is_empty() {
            return String::new();
        }
        // A heading is a sentence of its own: the period makes split_sentences
        // give it its own chunk, so it lands with a pause instead of running
        // into the paragraph below it.
        if body.ends_with(['.', '!', '?', ':', ';', ',']) {
            return body.to_string();
        }
        return format!("{body}.");
    }

    // Unordered list marker. A *run* of markers, not just one: nested bullets are
    // often written `** item` / `*** item`, and that is the case that prompted this
    // whole pass — espeak reads each asterisk. The trailing-whitespace check is what
    // keeps `*emphasis*` and `-5 degrees` intact, and a line that is *only* markers
    // was already dropped as a horizontal rule above.
    let marker_run = match s.chars().next() {
        Some(m @ ('-' | '*' | '+')) => s.chars().take_while(|&c| c == m).count(),
        _ => 0,
    };
    if marker_run > 0 && s[marker_run..].chars().next().is_some_and(char::is_whitespace) {
        s = s[marker_run..].trim_start();
        // Task list checkbox, only ever valid right after the bullet.
        for box_ in ["[ ] ", "[x] ", "[X] ", "[ ]", "[x]", "[X]"] {
            if let Some(rest) = s.strip_prefix(box_) {
                s = rest.trim_start();
                break;
            }
        }
    }

    // Footnote definition label (`[^1]: text`) — keep the text, drop the label.
    if let Some(rest) = s.strip_prefix("[^")
        && let Some(close) = rest.find("]:")
    {
        s = rest[close + 2..].trim_start();
    }

    s.to_string()
}

/// Remove inline markup from one line's text. Emphasis and escapes are *not* done
/// here — they run per paragraph in [`finish_paragraph`], since they can span a
/// line break. Everything below is line-scoped on purpose: an unclosed backtick or
/// bracket must not swallow the following line.
///
/// Order matters: images before links (an image is a link with a `!`), links
/// before code (a URL may contain backticks).
fn strip_inline(line: &str) -> String {
    let s = strip_images(line);
    let s = strip_links(&s);
    let s = strip_code_spans(&s);
    strip_html(&s)
}

/// `![alt](url)` and `![alt][ref]` — dropped whole, alt text included.
fn strip_images(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '!'
            && chars.get(i + 1) == Some(&'[')
            && let Some((_, end)) = bracketed_link(&chars, i + 1)
        {
            i = end;
            continue;
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// `[text](url)` / `[text][ref]` -> `text`; `<https://…>` autolinks are dropped.
fn strip_links(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '['
            && let Some((text, end)) = bracketed_link(&chars, i)
        {
            out.push_str(&text);
            i = end;
            continue;
        }
        if chars[i] == '<'
            && let Some(close) = chars[i..].iter().position(|&c| c == '>')
        {
            let inner: String = chars[i + 1..i + close].iter().collect();
            if inner.starts_with("http://")
                || inner.starts_with("https://")
                || inner.contains('@') && !inner.contains(' ')
            {
                i += close + 1;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// Match `[text](…)` or `[text][…]` starting at `open` (which must be a `[`).
/// Returns the link text and the index just past the whole construct.
fn bracketed_link(chars: &[char], open: usize) -> Option<(String, usize)> {
    let mut depth = 0usize;
    let mut close = None;
    for (i, &c) in chars.iter().enumerate().skip(open) {
        match c {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    let close = close?;
    let text: String = chars[open + 1..close].iter().collect();
    // The target must follow immediately, otherwise this is a plain bracketed
    // aside (`[sic]`) that should be left alone.
    let target_close = match chars.get(close + 1) {
        Some('(') => ')',
        Some('[') => ']',
        _ => return None,
    };
    let end = chars[close + 1..].iter().position(|&c| c == target_close)?;
    Some((text, close + 1 + end + 1))
}

/// `` `code` `` and ``` ``code`` ``` -> `code`. The content is kept: inline code
/// is usually a function or file name that belongs in the sentence.
fn strip_code_spans(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '`' {
            let ticks = chars[i..].iter().take_while(|&&c| c == '`').count();
            // Find the matching run of the same length.
            let mut j = i + ticks;
            let mut closed = None;
            while j < chars.len() {
                if chars[j] == '`' {
                    let run = chars[j..].iter().take_while(|&&c| c == '`').count();
                    if run == ticks {
                        closed = Some(j);
                        break;
                    }
                    j += run;
                    continue;
                }
                j += 1;
            }
            if let Some(j) = closed {
                out.extend(chars[i + ticks..j].iter());
                i = j + ticks;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// Remove emphasis runs (`**`, `__`, `*`, `_`, `~~`).
///
/// A run is only removed when it *hugs a word*: an opener is followed by a
/// non-space, a closer is preceded by a non-space, and the two must pair up. That
/// guard is what leaves arithmetic (`3 * 4 = 12`) and separators (`a _ b`) alone.
/// `_` additionally requires a word boundary on the outside, so `snake_case_name`
/// survives — fusing it into `snakecasename` would be worse than the underscore.
fn strip_emphasis(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    // Pass 1: find the runs and decide which ones pair into emphasis.
    let mut runs: Vec<(usize, usize, char)> = Vec::new(); // (start, len, marker)
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if matches!(c, '*' | '_' | '~') {
            let len = chars[i..].iter().take_while(|&&x| x == c).count();
            runs.push((i, len, c));
            i += len;
            continue;
        }
        i += 1;
    }

    let mut drop: Vec<(usize, usize)> = Vec::new();
    let mut open: Vec<usize> = Vec::new(); // indices into `runs`
    for (idx, &(start, len, marker)) in runs.iter().enumerate() {
        if marker == '~' && len != 2 {
            continue; // a lone ~ is not strikethrough
        }
        let before = start.checked_sub(1).map(|p| chars[p]);
        let after = chars.get(start + len).copied();
        let can_open = after.is_some_and(|c| !c.is_whitespace());
        let can_close = before.is_some_and(|c| !c.is_whitespace());
        // Intra-word `_` (snake_case) is not emphasis in CommonMark either, and
        // removing it would fuse `snake_case_name` into one unreadable word.
        let intra_word = marker == '_'
            && before.is_some_and(|c| c.is_alphanumeric())
            && after.is_some_and(|c| c.is_alphanumeric());
        if intra_word {
            continue;
        }
        // Closing a run of the same marker and length wins over opening a new one.
        if can_close
            && let Some(pos) = open
                .iter()
                .rposition(|&o| runs[o].2 == marker && runs[o].1 == len && o != idx)
        {
            let opener = open.remove(pos);
            drop.push((runs[opener].0, runs[opener].1));
            drop.push((start, len));
            continue;
        }
        if can_open {
            open.push(idx);
            continue;
        }
        // Whitespace on both sides and 2+ markers: not arithmetic, not a separator,
        // just an orphaned delimiter (`** bold **` writes one at each end). A *lone*
        // `*` or `_` in that position is left alone — it may be `3 * 4`.
        if !can_close && len >= 2 {
            drop.push((start, len));
        }
    }

    if drop.is_empty() {
        return s.to_string();
    }
    let mut skip = vec![false; chars.len()];
    for (start, len) in drop {
        for f in skip.iter_mut().skip(start).take(len) {
            *f = true;
        }
    }
    chars.iter().enumerate().filter(|(i, _)| !skip[*i]).map(|(_, &c)| c).collect()
}

/// `\*` -> `*`. A backslash escape meant the literal character all along, and the
/// backslash itself would otherwise be pronounced.
fn strip_escapes(s: &str) -> String {
    const ESCAPABLE: &[char] =
        &['*', '_', '`', '[', ']', '#', '~', '\\', '(', ')', '<', '>', '|', '!', '.'];
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' && chars.peek().is_some_and(|n| ESCAPABLE.contains(n)) {
            continue;
        }
        out.push(c);
    }
    out
}

/// Drop raw HTML tags and comments (`<br>`, `<details>`, `<!-- … -->`).
///
/// The tag name must be one we actually recognize. Accepting any identifier looked
/// tempting and was wrong: it ate `Vec<String>` and `Option<PathBuf>` out of this
/// project's own docs, and a swallowed type parameter is far worse than a `<foo>`
/// getting pronounced.
fn strip_html(s: &str) -> String {
    if !s.contains('<') {
        return s.to_string();
    }
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '<'
            && let Some(close) = chars[i..].iter().position(|&c| c == '>')
        {
            let inner: String = chars[i + 1..i + close].iter().collect();
            if is_html_tag(&inner) {
                i += close + 1;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// The inside of a `<…>`: an HTML comment, or a known tag name (optionally closing,
/// optionally self-closing, optionally carrying attributes).
fn is_html_tag(inner: &str) -> bool {
    /// Block/inline tags that actually turn up in markdown prose. Deliberately not
    /// exhaustive — an unlisted tag is read aloud, which is the safe failure.
    const TAGS: &[&str] = &[
        "a", "abbr", "b", "blockquote", "br", "caption", "center", "cite", "code", "dd",
        "details", "div", "dl", "dt", "em", "figcaption", "figure", "font", "h1", "h2", "h3",
        "h4", "h5", "h6", "hr", "i", "img", "kbd", "li", "mark", "ol", "p", "picture", "pre",
        "q", "s", "samp", "small", "span", "strong", "sub", "summary", "sup", "table", "tbody",
        "td", "tfoot", "th", "thead", "tr", "u", "ul", "var", "video",
    ];
    if inner.starts_with("!--") {
        return true;
    }
    // Real HTML never puts a space after `<`, but a comparison does: `a < b > c`
    // must survive, and `b` happens to be a tag name.
    if inner.starts_with(char::is_whitespace) {
        return false;
    }
    let inner = inner.trim_start_matches('/').trim_end_matches('/').trim();
    let name = inner.split([' ', '\t', '=']).next().unwrap_or("");
    TAGS.contains(&name.to_ascii_lowercase().as_str())
}


/// Stripping rules, from the angle that matters here: what does the listener hear?
/// Every case is a line that used to be read aloud as punctuation, or a line that
/// must survive untouched because removing its markers would garble real words.
#[cfg(test)]
mod tests {
    use super::strip;

    /// The reported bug: nested bullets written as a run of asterisks, where espeak
    /// pronounced each one. Both the marker run and spaced-out emphasis must go.
    #[test]
    fn nested_bullet_runs_are_not_spoken() {
        assert_eq!(strip("** deeply nested item"), "deeply nested item");
        assert_eq!(strip("  *** third level"), "third level");
        assert_eq!(strip("  ** spaced emphasis **"), "spaced emphasis");
        assert_eq!(strip("-- dashed nesting"), "dashed nesting");
    }

    /// Ordinary emphasis, the well-formed kind.
    #[test]
    fn emphasis_markers_are_not_spoken() {
        assert_eq!(strip("  - **deeply nested** item"), "deeply nested item");
        assert_eq!(strip("    * *one* and __two__ and ~~three~~"), "one and two and three");
        assert_eq!(strip("+ plain bullet"), "plain bullet");
    }

    /// The false-positive guard. These are prose/code, not markup, and any of them
    /// coming out mangled is worse than leaving a stray asterisk in.
    #[test]
    fn arithmetic_and_identifiers_survive() {
        for text in ["3 * 4 = 12", "snake_case_name", "a _ b", "5 - 3 is two", "-5 degrees"] {
            assert_eq!(strip(text), text, "{text:?} must pass through unchanged");
        }
    }

    /// Headings lose their hashes and gain a period, so each lands as its own
    /// chunk with a pause instead of running into the paragraph below.
    #[test]
    fn headings_become_sentences() {
        assert_eq!(strip("## A heading"), "A heading.");
        assert_eq!(strip("### Closed heading ###"), "Closed heading.");
        assert_eq!(strip("# Already done."), "Already done.");
        assert_eq!(strip("#hashtag not a heading"), "#hashtag not a heading");
    }

    /// Fenced code is unlistenable; the prose around it must still be spoken.
    #[test]
    fn fenced_code_disappears_but_prose_stays() {
        let doc = "Before the block.\n```rust\nlet x = 1;\n```\nAfter the block.";
        assert_eq!(strip(doc), "Before the block.\nAfter the block.");
        let tilde = "Before.\n~~~\ncode ``` here\n~~~\nAfter.";
        assert_eq!(strip(tilde), "Before.\nAfter.");
    }

    /// Inline code is usually a function or file name and belongs in the sentence,
    /// so only the backticks go.
    #[test]
    fn inline_code_keeps_its_content() {
        assert_eq!(strip("Call `Pipeline::new` first."), "Call Pipeline::new first.");
        assert_eq!(strip("Use ``a ` b`` here."), "Use a ` b here.");
    }

    /// Link text is the readable half; URLs are not. Images have no readable half.
    #[test]
    fn links_read_their_text_and_images_vanish() {
        assert_eq!(strip("See [playback doc](docs/playback_control.md) now."), "See playback doc now.");
        assert_eq!(strip("A [ref link][1] here."), "A ref link here.");
        assert_eq!(strip("![a diagram](img.png)"), "");
        assert_eq!(strip("text ![a diagram](img.png) more"), "text more");
        assert_eq!(strip("[1]: https://example.com"), "");
        assert_eq!(strip("Go to <https://example.com> now."), "Go to now.");
        // A bracketed aside is not a link and keeps its brackets.
        assert_eq!(strip("that word [sic] there"), "that word [sic] there");
    }

    /// Tables and rules read linearly are pure punctuation.
    #[test]
    fn tables_and_rules_are_dropped() {
        let doc = "Intro.\n| a | b |\n|---|---|\n| 1 | 2 |\nOutro.";
        assert_eq!(strip(doc), "Intro.\nOutro.");
        assert_eq!(strip("---"), "");
        assert_eq!(strip("* * *"), "");
        assert_eq!(strip("Title\n====="), "Title");
    }

    /// Blockquotes, task boxes and footnote labels are structure, not speech.
    #[test]
    fn quotes_checkboxes_and_footnotes_lose_their_markers() {
        assert_eq!(strip("> > quoted text"), "quoted text");
        assert_eq!(strip("- [ ] todo item"), "todo item");
        assert_eq!(strip("- [x] done item"), "done item");
        assert_eq!(strip("[^1]: the footnote body"), "the footnote body");
    }

    /// Ordered markers stay: the number is information you want when listening.
    #[test]
    fn ordered_list_numbers_are_kept() {
        assert_eq!(strip("1. first step"), "1. first step");
        assert_eq!(strip("2) second step"), "2) second step");
    }

    /// Raw HTML is invisible when rendered, so it must be inaudible too — but
    /// a comparison in prose is not a tag.
    #[test]
    fn html_tags_go_but_generics_and_comparisons_stay() {
        assert_eq!(strip("line<br>break"), "linebreak");
        assert_eq!(strip("<details><summary>x</summary>"), "x");
        assert_eq!(strip("<!-- hidden -->"), "");
        assert_eq!(strip("a < b > c"), "a < b > c");
        // Accepting any identifier as a tag ate type parameters out of this
        // project's own docs — the exact regression this pins.
        assert_eq!(strip("Vec<String> and Option<PathBuf>."), "Vec<String> and Option<PathBuf>.");
        assert_eq!(strip("run `bench.sh <label>` now"), "run bench.sh <label> now");
    }

    /// A backslash-escaped marker meant the literal character.
    #[test]
    fn escapes_yield_the_literal_character() {
        assert_eq!(strip("a \\* b"), "a * b");
    }

    /// Malformed input is common in real notes and must not panic or eat the file.
    #[test]
    fn unterminated_constructs_are_handled() {
        assert_eq!(strip("Before.\n```\nnever closed"), "Before.");
        assert_eq!(strip("an *unclosed emphasis"), "an *unclosed emphasis");
        assert_eq!(strip("a `unclosed code"), "a `unclosed code");
    }

    /// Authors wrap `**bold phrases**` across a line break; a line-local pass left
    /// the orphaned markers behind, which is exactly what gets read aloud.
    #[test]
    fn emphasis_pairs_across_a_line_break() {
        assert_eq!(strip("they are **not shipped\nwith the repo**."), "they are not shipped\nwith the repo.");
        // ...but not across a paragraph break, where an unrelated `*` could match.
        assert_eq!(strip("a **one\n\ntwo** b"), "a **one\ntwo** b");
    }

    /// Running the pass twice must not keep eating characters — `--send` and the
    /// one-shot path both strip, and a doc may be re-fed through either.
    #[test]
    fn stripping_is_idempotent() {
        let doc = "# Title\n\n- **bold** item with `code` and [a link](u)\n\n```\nx\n```\n\n| a |\n\nEnd.";
        let once = strip(doc);
        assert_eq!(strip(&once), once);
    }
}
