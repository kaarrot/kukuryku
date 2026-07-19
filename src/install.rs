//! `ryk --install-assets`: download the pre-split Kokoro assets from the GitHub
//! release and unpack them next to the running binary, so an installed `ryk`
//! finds its models from any working directory.
//!
//! Replaces a bash fetch script: same tag, same zip, same sha256, but it runs
//! wherever `ryk` runs — including Windows PowerShell, which has no bash.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

use crate::kokoro::{env_or, exe_assets_dir, user_assets_dir};

/// Pinned deliberately, not `latest`: the assets are versioned separately from
/// the code (they only change if the Kokoro weights or split_kokoro.py do), and
/// /releases/latest/ would point at the first code release published without the
/// zip attached, 404ing this. Bump when a new asset release goes up.
pub const ASSET_TAG: &str = "kokoro-onyx-model";
pub const ASSET_REPO: &str = "kaarrot/kukuryku";
pub const ASSET_FILE: &str = "kokoro-onyx.zip";
/// sha256 of the zip on the pinned tag above; re-zipping the assets changes it.
pub const ASSET_SHA256: &str = "469f4a2425a57454bddb93cbe4dfdb6628f8f1de3a9d85fe6193f77e258de594";

/// Where `--install-assets` writes the bundle by default: the OS-specific
/// per-user data dir (see [`crate::kokoro::user_assets_dir`]). Matches the
/// second (and now primary) lookup arm of [`crate::kokoro::local_assets_dir`]
/// so what we write is what the runtime finds.
pub fn install_dir() -> Result<PathBuf> {
    user_assets_dir().context(
        "no user data directory available on this platform \
         (set KOKORO_TRACT_DIR or pass --dev to install beside the binary)",
    )
}

/// Dev install location: `kokoro-onyx/` next to the running executable.
/// Selected by `--install-assets --dev` (or `KUKURYKU_ASSET_DIR=exe`) so a
/// `cargo run` on a checkout doesn't drop 600 MB into the real user data dir.
pub fn dev_install_dir() -> Result<PathBuf> {
    exe_assets_dir().context("locating the running executable")
}

/// Which install target to write to, resolved from CLI + env in this order:
/// 1. `KUKURYKU_ASSET_DIR` — explicit absolute path (or `exe` for the exe-adjacent dir).
/// 2. `dev` flag (from `--install-assets --dev`) — force the exe-adjacent dir.
/// 3. Default: the user data dir.
fn resolve_install_dir(dev: bool) -> Result<PathBuf> {
    if let Some(v) = std::env::var_os("KUKURYKU_ASSET_DIR") {
        if v == "exe" {
            return dev_install_dir();
        }
        return Ok(PathBuf::from(v));
    }
    if dev { dev_install_dir() } else { install_dir() }
}

/// `--install-assets` entry point. `dev=true` (from `--dev`) installs beside
/// the running binary instead of the user data dir — matches the pre-XDG
/// layout, useful when iterating on a `cargo run` checkout without polluting
/// the real `~/.local/share/kukuryku`.
pub fn run(dev: bool) -> Result<()> {
    let dest = resolve_install_dir(dev)?;
    // The zip has a top-level kokoro-onyx/, so extracting at the parent lands the
    // files in `dest` itself rather than dest/kokoro-onyx/.
    let root = dest
        .parent()
        .context("install dir has no parent")?
        .to_path_buf();

    if dest.join("stage1.onnx").is_file() && dest.join("stage2.onnx").is_file() {
        println!("assets already present at {}", dest.display());
        println!("delete that directory to re-install.");
        return Ok(());
    }

    fs::create_dir_all(&root)
        .with_context(|| format!("creating {}", root.display()))?;

    let repo = env_or("KUKURYKU_REPO", ASSET_REPO);
    let tag = env_or("KUKURYKU_ASSET_TAG", ASSET_TAG);
    let url = format!("https://github.com/{repo}/releases/download/{tag}/{ASSET_FILE}");

    // Download inside the app's own OS-specific dir (parent of the extract
    // target, i.e. `dirs::data_dir()/kukuryku/`) rather than /tmp: that keeps
    // the ~600 MB fetch on the same filesystem as the final extract, so the
    // extract is a plain rename/copy within one fs instead of a cross-device
    // copy from tmpfs. Cleanup happens via the RAII guard below on every exit.
    let part = root.join(format!("{ASSET_FILE}.part"));

    // Ensure the zip is removed on every early return below (mismatch, extract
    // failure, missing marker, panic) — not just the success path.
    struct TempFile(PathBuf);
    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
        }
    }
    let _tmp_guard = TempFile(part.clone());

    let digest = download(&url, &part)?;

    // Only enforce the pinned hash for the pinned tag: an overridden tag/repo is
    // some other zip, and this hash says nothing about it.
    if tag == ASSET_TAG && repo == ASSET_REPO && digest != ASSET_SHA256 {
        bail!("sha256 mismatch\n  expected {ASSET_SHA256}\n  got      {digest}");
    }

    println!("== extracting -> {} ==", dest.display());
    unzip(&part, &root)?;

    if !dest.join("stage1.onnx").is_file() {
        bail!("extracted archive has no {}/stage1.onnx", dest.display());
    }
    println!("done — try:  ryk \"Hello world.\"");
    Ok(())
}

/// Stream the download to `part`, hashing as it goes: the zip is ~576 MB, so it
/// never lands in memory whole.
fn download(url: &str, part: &Path) -> Result<String> {
    println!("== downloading ==");
    println!("   {url}");

    let resp = ureq::get(url)
        .call()
        .with_context(|| format!("fetching {url}"))?;
    let total: u64 = resp
        .header("Content-Length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let mut reader = resp.into_reader();
    let mut file = fs::File::create(part)
        .with_context(|| format!("creating {}", part.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024];
    let mut done: u64 = 0;
    let mut last_pct = u64::MAX;

    loop {
        let n = reader.read(&mut buf).context("reading response body")?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        file.write_all(&buf[..n])
            .with_context(|| format!("writing {}", part.display()))?;
        done += n as u64;
        if total > 0 {
            let pct = done * 100 / total;
            if pct != last_pct {
                print!("\r   {pct}% ({} MB / {} MB)", done >> 20, total >> 20);
                let _ = std::io::stdout().flush();
                last_pct = pct;
            }
        }
    }
    println!();
    file.sync_all().ok();

    Ok(hasher.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

fn unzip(zip_path: &Path, root: &Path) -> Result<()> {
    let file = fs::File::open(zip_path)
        .with_context(|| format!("opening {}", zip_path.display()))?;
    let mut archive = zip::ZipArchive::new(file).context("reading zip archive")?;

    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).context("reading zip entry")?;
        // `enclosed_name` rejects absolute paths and `..` traversal; a zip that
        // has them is malformed and we'd rather fail than write outside `root`.
        let rel = entry
            .enclosed_name()
            .with_context(|| format!("unsafe path in archive: {}", entry.name()))?;
        let out = root.join(rel);

        if entry.is_dir() {
            fs::create_dir_all(&out)
                .with_context(|| format!("creating {}", out.display()))?;
            continue;
        }
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        let mut sink = fs::File::create(&out)
            .with_context(|| format!("creating {}", out.display()))?;
        std::io::copy(&mut entry, &mut sink)
            .with_context(|| format!("extracting {}", out.display()))?;
    }
    Ok(())
}
