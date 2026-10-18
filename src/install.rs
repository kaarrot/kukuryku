//! Fetch the pre-split Kokoro assets from the GitHub release and unpack them into
//! the OS-specific per-user data dir, so `ryk` finds its models from any working
//! directory, with or without a checkout.
//!
//! Two entry points share one code path: `ryk` itself calls [`ensure`] the first
//! time it needs the stages and finds none (see `kokoro::resolve_assets_tract`),
//! and `ryk --install-assets` calls [`run`] to do the same fetch ahead of time.
//!
//! Replaces a bash fetch script: same tag, same zip, same sha256, but it runs
//! wherever `ryk` runs — including Windows PowerShell, which has no bash.

use std::fs;
use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

use crate::kokoro::{env_or, exe_assets_dir, stages_ready, user_assets_dir};

/// Pinned deliberately, not `latest`: the assets are versioned separately from
/// the code (they only change if the Kokoro weights or split_kokoro.py do), and
/// /releases/latest/ would point at the first code release published without the
/// zip attached, 404ing this. Bump when a new asset release goes up.
pub const ASSET_TAG: &str = "kokoro-onyx-model";
pub const ASSET_REPO: &str = "kaarrot/kukuryku";
pub const ASSET_FILE: &str = "kokoro-onyx.zip";
/// sha256 of the zip on the pinned tag above; re-zipping the assets changes it.
pub const ASSET_SHA256: &str = "469f4a2425a57454bddb93cbe4dfdb6628f8f1de3a9d85fe6193f77e258de594";
/// Top-level directory inside the zip. Stripped on extract, so the bundle lands
/// in whatever directory the caller asked for, not `<dir>/../kokoro-onyx`.
const ARCHIVE_ROOT: &str = "kokoro-onyx";

/// Where `--install-assets` writes the bundle by default: the OS-specific
/// per-user data dir (see [`crate::kokoro::user_assets_dir`]). Matches the
/// first lookup arm of `kokoro::resolve_assets_tract`, so what we write is what
/// the runtime finds.
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
pub fn resolve_install_dir(dev: bool) -> Result<PathBuf> {
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
    if !ensure(&dest)? {
        println!("assets already present at {}", dest.display());
        println!("delete that directory to re-install.");
        return Ok(());
    }
    println!("done — try:  ryk \"Hello world.\"");
    Ok(())
}

/// Make sure `dest` holds `stage1.onnx` + `stage2.onnx`, downloading and
/// unpacking the release zip into it if not. Returns whether it downloaded.
///
/// Safe to call from several `ryk` processes at once (an editor firing twice on
/// a fresh box): the first takes a lock and downloads, the rest wait on it and
/// then find the stages in place. The lock is an OS file lock, so a killed
/// download releases it and the next run starts over.
pub fn ensure(dest: &Path) -> Result<bool> {
    if stages_ready(dest) {
        return Ok(false);
    }
    let name = dest
        .file_name()
        .context("install dir has no final component")?
        .to_string_lossy()
        .into_owned();
    let root = dest.parent().context("install dir has no parent")?;
    fs::create_dir_all(root).with_context(|| format!("creating {}", root.display()))?;

    let lock_path = root.join(format!("{name}.lock"));
    let lock = fs::File::create(&lock_path)
        .with_context(|| format!("creating {}", lock_path.display()))?;
    if lock.try_lock().is_err() {
        eprintln!("[ryk] another ryk is downloading the model into {}; waiting", dest.display());
        lock.lock().with_context(|| format!("locking {}", lock_path.display()))?;
    }
    // Whoever held the lock may have just finished the job.
    if stages_ready(dest) {
        return Ok(false);
    }

    let repo = env_or("KUKURYKU_REPO", ASSET_REPO);
    let tag = env_or("KUKURYKU_ASSET_TAG", ASSET_TAG);
    let url = format!("https://github.com/{repo}/releases/download/{tag}/{ASSET_FILE}");
    eprintln!("[ryk] model not found; downloading it once (~576 MB) into {}", dest.display());

    // Download beside the extract target rather than /tmp: that keeps the
    // ~600 MB fetch on the same filesystem as the bundle, and off a tmpfs that
    // may be too small for it. Removed by the guard on every return; a killed
    // process leaves it behind, and the next download truncates it.
    let part = root.join(format!("{name}.zip.part"));
    struct TempFile(PathBuf);
    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
        }
    }
    let _tmp_guard = TempFile(part.clone());

    let digest = download(&url, &part).with_context(|| {
        format!(
            "downloading the model. The first run needs {url} to be reachable. \
             On a machine without that, copy a kokoro-onyx/ directory \
             (stage1.onnx, stage2.onnx, voices/) to {}, or point KOKORO_TRACT_DIR at one",
            dest.display()
        )
    })?;

    // Only enforce the pinned hash for the pinned tag: an overridden tag/repo is
    // some other zip, and this hash says nothing about it.
    if tag == ASSET_TAG && repo == ASSET_REPO && digest != ASSET_SHA256 {
        bail!("sha256 mismatch\n  expected {ASSET_SHA256}\n  got      {digest}");
    }

    eprintln!("[ryk] extracting -> {}", dest.display());
    unzip(&part, dest)?;

    if !stages_ready(dest) {
        bail!("extracted archive has no stage1.onnx + stage2.onnx in {}", dest.display());
    }
    // Safe while others wait on it: each re-checks the stages once it gets the
    // lock, and a newcomer sees them before it ever creates a lock file.
    let _ = fs::remove_file(&lock_path);
    Ok(true)
}

/// Stream the download to `part`, hashing as it goes: the zip is ~576 MB, so it
/// never lands in memory whole. Progress goes to stderr: a `\r`-updated percent
/// on a terminal, one line per 10% otherwise (an editor or daemon log).
fn download(url: &str, part: &Path) -> Result<String> {
    eprintln!("   {url}");

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
    let tty = std::io::stderr().is_terminal();
    let step = if tty { 1 } else { 10 };
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
            let pct = done * 100 / total / step * step;
            if pct != last_pct {
                let line = format!("   {pct}% ({} MB / {} MB)", done >> 20, total >> 20);
                if tty {
                    eprint!("\r{line}");
                } else {
                    eprintln!("{line}");
                }
                last_pct = pct;
            }
        }
    }
    if tty {
        eprintln!();
    }
    file.sync_all().ok();

    Ok(hasher.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// Unpack `zip_path` into `dest`, dropping the archive's top-level
/// `kokoro-onyx/` so the files land in `dest` whatever it is called.
///
/// Built to be interrupted (Ctrl-C, Android killing Termux): each file is written
/// to `<file>.part` and renamed into place, and the two stages go last. So
/// `stage1.onnx` + `stage2.onnx` both existing means every file is complete,
/// which is the one check the runtime makes before it trusts the dir.
fn unzip(zip_path: &Path, dest: &Path) -> Result<()> {
    let file = fs::File::open(zip_path)
        .with_context(|| format!("opening {}", zip_path.display()))?;
    let mut archive = zip::ZipArchive::new(file).context("reading zip archive")?;

    let mut order: Vec<(bool, usize)> = Vec::with_capacity(archive.len());
    for i in 0..archive.len() {
        let entry = archive.by_index(i).context("reading zip entry")?;
        let is_stage = entry
            .enclosed_name()
            .and_then(|p| p.file_name().map(|n| n == "stage1.onnx" || n == "stage2.onnx"))
            .unwrap_or(false);
        order.push((is_stage, i));
    }
    // `false < true`: everything else first, then the stages.
    order.sort();

    for (_, i) in order {
        let mut entry = archive.by_index(i).context("reading zip entry")?;
        // `enclosed_name` rejects absolute paths and `..` traversal; a zip that
        // has them is malformed and we'd rather fail than write outside `dest`.
        let rel = entry
            .enclosed_name()
            .with_context(|| format!("unsafe path in archive: {}", entry.name()))?;
        let rel = rel.strip_prefix(ARCHIVE_ROOT).unwrap_or(&rel);
        let out = dest.join(rel);

        if entry.is_dir() {
            fs::create_dir_all(&out)
                .with_context(|| format!("creating {}", out.display()))?;
            continue;
        }
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        let mut tmp_name = out.file_name().context("zip entry has no file name")?.to_os_string();
        tmp_name.push(".part");
        let tmp = out.with_file_name(tmp_name);
        let mut sink = fs::File::create(&tmp)
            .with_context(|| format!("creating {}", tmp.display()))?;
        std::io::copy(&mut entry, &mut sink)
            .with_context(|| format!("extracting {}", out.display()))?;
        sink.sync_all().ok();
        drop(sink);
        fs::rename(&tmp, &out)
            .with_context(|| format!("moving {} into place", out.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use zip::write::SimpleFileOptions;

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ryk-install-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A zip shaped like the release one: everything under `kokoro-onyx/`, stages
    /// listed first so the reordering in `unzip` has something to do.
    fn bundle_zip(path: &Path) {
        let mut zw = zip::ZipWriter::new(fs::File::create(path).unwrap());
        let opts = SimpleFileOptions::default();
        zw.add_directory("kokoro-onyx/", opts).unwrap();
        for (name, body) in [
            ("kokoro-onyx/stage1.onnx", "s1"),
            ("kokoro-onyx/stage2.onnx", "s2"),
            ("kokoro-onyx/model.onnx", "m"),
            ("kokoro-onyx/voices/af_heart.bin", "v"),
        ] {
            zw.start_file(name, opts).unwrap();
            zw.write_all(body.as_bytes()).unwrap();
        }
        zw.finish().unwrap();
    }

    /// The archive's `kokoro-onyx/` is dropped, so a target with another name
    /// (KUKURYKU_ASSET_DIR, KOKORO_TRACT_DIR) gets the files directly.
    #[test]
    fn unzip_lands_in_dest_whatever_its_name() {
        let tmp = scratch("name");
        let zip = tmp.join("b.zip");
        bundle_zip(&zip);
        let dest = tmp.join("models");
        unzip(&zip, &dest).unwrap();

        assert!(stages_ready(&dest));
        assert_eq!(fs::read_to_string(dest.join("voices/af_heart.bin")).unwrap(), "v");
        assert!(!tmp.join("kokoro-onyx").exists());
        let _ = fs::remove_dir_all(&tmp);
    }

    /// No `.part` file survives a completed extract.
    #[test]
    fn unzip_leaves_no_part_files() {
        let tmp = scratch("part");
        let zip = tmp.join("b.zip");
        bundle_zip(&zip);
        let dest = tmp.join("kokoro-onyx");
        unzip(&zip, &dest).unwrap();

        for dir in [dest.clone(), dest.join("voices")] {
            for entry in fs::read_dir(&dir).unwrap() {
                let name = entry.unwrap().file_name();
                assert!(!name.to_string_lossy().ends_with(".part"), "{name:?} left behind");
            }
        }
        let _ = fs::remove_dir_all(&tmp);
    }

    /// A dir that already has both stages is left alone: no lock, no network.
    #[test]
    fn ensure_is_a_noop_when_stages_exist() {
        let tmp = scratch("noop");
        let dest = tmp.join("kokoro-onyx");
        fs::create_dir_all(&dest).unwrap();
        fs::write(dest.join("stage1.onnx"), "s1").unwrap();
        fs::write(dest.join("stage2.onnx"), "s2").unwrap();

        assert!(!ensure(&dest).unwrap());
        assert!(!tmp.join("kokoro-onyx.lock").exists());
        let _ = fs::remove_dir_all(&tmp);
    }
}
