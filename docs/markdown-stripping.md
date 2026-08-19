# Markdown stripping (`src/markdown.rs`)

## Why

espeak-ng pronounces punctuation. Feeding it a `.md` file meant hearing the syntax as well as
the prose — the case that prompted this was a nested bullet written `** deeply nested item`,
where each asterisk in the marker run got read out. Tables, code fences and heading hashes are
the same problem at larger scale.

Nothing in the pipeline normalized text before this: `read_text()` trimmed the input and handed
it straight to `split_sentences()` → `phonemize()`. The vocab filter in `prepare_from_phonemes`
only drops *phonemes* the model lacks — by then espeak has already turned `**` into speech.

So: a cleanup pass over the whole document, **on by default**, `--raw` / `KOKORO_RAW=1` to
bypass it. `ryk --show-text` prints the result and exits, which is how you check it on a real
file without listening to the whole thing.

## Where it runs

`kokoro::clean_input` is called from the two text readers — `kokoro::read_text` (one-shot) and
`serve::client_text` (`--send`). Both are argv-side, so `--raw` is honoured per invocation and
the daemon wire header stays `voice\tlang\tspeed`; the daemon speaks the body it receives and
never re-strips it.

It must run on the **whole document, before `split_sentences`**: the fenced-code state machine
is multi-line, and emphasis is resolved per paragraph (see below).

## Block rules

Applied per line, in this order. Anything dropped wholesale also ends the current paragraph.

| Input | Result |
|---|---|
| ` ```lang ` / `~~~` fences and everything between | dropped (only the same marker closes; an unterminated fence runs to EOF, as in any renderer) |
| line whose trimmed form starts with `\|` | dropped — table rows and the `\|---\|` separator make no sense read linearly |
| 3+ of `-` `*` `_` `=` and nothing else | dropped (horizontal rule, setext underline) |
| `[ref]: https://…` | dropped — a definition line, no prose in it |
| `### Heading ###` | `Heading.` — closing hashes dropped, trailing period added so `split_sentences` gives it its own chunk and a pause |
| `>` / `> >` blockquote prefix | removed, text kept |
| a **run** of `-` `*` `+` then whitespace | removed — `** nested`, `*** deeper` are the nested-bullet style that started this |
| `1.` / `1)` ordered marker | **kept** — the number is information when you're listening rather than looking |
| `[ ]` / `[x]` right after a bullet | removed |
| `[^1]: text` footnote definition | label removed, body kept |

## Inline rules

Images → links → code spans → HTML run **per line** (an unclosed backtick or bracket must not
swallow the following line). Emphasis and escapes run **per paragraph**, because authors wrap
`**bold phrases**` across a line break and a line-local pass would leave the orphaned markers
to be spoken. A paragraph is also the right upper bound: pairing across a whole document could
match a stray `*` on line 5 with an unrelated one 400 lines later.

| Input | Result |
|---|---|
| `![alt](url)`, `![alt][ref]` | dropped whole, alt text included |
| `[text](url)`, `[text][ref]` | `text` |
| `<https://…>` autolink | dropped |
| `` `code` ``, ``` ``code`` ``` | `code` — content kept: inline code is usually a function or file name that belongs in the sentence |
| `**b**` `__b__` `*i*` `_i_` `~~s~~` | markers removed |
| an isolated run of 2+ markers | removed — the orphan each end of a spaced `** bold **` |
| `<br>`, `<details>`, `<!-- … -->` | dropped |
| `\*` | `*` |
| line that is now empty | dropped |
| runs of whitespace left by a removal | collapsed to one space |

## False-positive guards

A false positive silently mangles speech, which is worse than leaving a stray marker in. Each
of these is pinned by a test in `src/markdown.rs`:

- **Emphasis must hug a word.** An opener is followed by a non-space, a closer preceded by one,
  and the two must pair. That leaves `3 * 4 = 12` and `a _ b` alone. Only a run of *2 or more*
  markers is removed when isolated by whitespace — a lone `*` there may be arithmetic.
- **Intra-word `_` is not emphasis.** `snake_case_name` survives; fusing it into
  `snakecasename` would be worse than pronouncing the underscore. (CommonMark agrees.)
- **HTML needs a known tag name, and no space after `<`.** Accepting any identifier ate
  `Vec<String>` and `Option<PathBuf>` out of this project's own docs; requiring no leading space
  keeps `a < b > c`. The tag list in `is_html_tag` is deliberately not exhaustive — an unlisted
  tag gets read aloud, which is the safe failure.
- **A bullet marker needs trailing whitespace.** Keeps `*emphasis*` and `-5 degrees` intact.
- **`[sic]` is not a link.** A bracket group only counts when `(`/`[` follows immediately.

## Not handled, on purpose

**4-space indented code blocks.** They are indistinguishable from a wrapped list continuation,
so dropping them would silently eat prose. Fenced blocks cover the common case.

## Checking it

```bash
cargo test                                  # 22 unit tests, no espeak-ng or assets needed
ryk --show-text < notes.md                  # what would be spoken
ryk --show-text --raw < notes.md            # unchanged, for comparison
```

The pass was validated against this repo's own five markdown files: no residual `**`, fences,
table pipes or link URLs in the output, and no prose words lost apart from URL components.
