//! Minimal HuggingFace Hub file download into the standard HF cache.
//!
//! Rust port of the former pure-Python `gigatoken._load.hub`: same endpoint
//! and URL layout as `huggingface_hub.hf_hub_download`, same token discovery
//! (HF_TOKEN env var, then the token file written by `hf auth login`), and
//! the same cache directory resolution, without requiring huggingface_hub,
//! tokenizers, or transformers. Files already present in the standard HF
//! cache are served with a pure-filesystem lookup (no network); on a miss the
//! file is downloaded straight into the shared cache — under the commit hash
//! the Hub reports via `x-repo-commit`, with the branch ref recorded — so
//! huggingface_hub and later lookups serve it from the same place.

use eyre::{Context, Result, bail};
use std::fmt;
use std::io;
use std::path::PathBuf;

/// Filename suffixes of local tokenizer files (tokenizer.json contents and
/// raw sentencepiece models — the formats the hf loader reads from disk).
/// A name ending in one of these is never treated as a Hub repo id, so a
/// mistyped local path fails fast instead of hitting the network. Keep in
/// sync with `gigatoken._load.hub.TOKENIZER_FILE_SUFFIXES`.
pub const TOKENIZER_FILE_SUFFIXES: &[&str] = &[".json", ".model"];

/// Whether `name` is shaped like a HuggingFace Hub repo id: `org/name`, or a
/// bare legacy repo name like `gpt2`. At most one slash, and not something
/// that is obviously a filesystem path to a local tokenizer file.
pub fn looks_like_repo_id(name: &str) -> bool {
    let word = |c: char| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-');
    let part_ok = |part: &str, first_alnum: bool| {
        let mut chars = part.chars();
        let Some(first) = chars.next() else {
            return false;
        };
        let first_ok = if first_alnum { first.is_ascii_alphanumeric() } else { word(first) };
        first_ok && chars.all(word)
    };
    let mut parts = name.split('/');
    let (org, rest) = (parts.next().unwrap_or(""), parts.next());
    parts.next().is_none()
        && part_ok(org, true)
        && rest.is_none_or(|r| part_ok(r, false))
        && !TOKENIZER_FILE_SUFFIXES.iter().any(|s| name.ends_with(s))
}

fn env_nonempty(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.is_empty())
}

/// $HF_HOME, defaulting to $XDG_CACHE_HOME/huggingface then
/// ~/.cache/huggingface — the root for both the hub cache and the token file.
fn hf_home() -> PathBuf {
    env_nonempty("HF_HOME").map(PathBuf::from).unwrap_or_else(|| {
        env_nonempty("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::home_dir().expect("home dir").join(".cache"))
            .join("huggingface")
    })
}

/// The standard HuggingFace hub cache directory, resolved like
/// huggingface_hub does it: HF_HUB_CACHE, then $HF_HOME/hub, then
/// $XDG_CACHE_HOME/huggingface/hub, then ~/.cache/huggingface/hub.
pub fn hf_hub_cache_dir() -> PathBuf {
    env_nonempty("HF_HUB_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| hf_home().join("hub"))
}

/// The HuggingFace access token, discovered like huggingface_hub does it:
/// the HF_TOKEN (or legacy HUGGING_FACE_HUB_TOKEN) environment variable,
/// then the token file (HF_TOKEN_PATH, default $HF_HOME/token).
pub fn get_hf_token() -> Option<String> {
    if let Some(token) = env_nonempty("HF_TOKEN")
        .or_else(|| env_nonempty("HUGGING_FACE_HUB_TOKEN"))
        .map(|t| t.trim().to_owned())
        .filter(|t| !t.is_empty())
    {
        return Some(token);
    }
    let token_path = env_nonempty("HF_TOKEN_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| hf_home().join("token"));
    let token = std::fs::read_to_string(token_path).ok()?;
    let token = token.trim();
    (!token.is_empty()).then(|| token.to_owned())
}

/// A full git commit hash: cache snapshot directories are named by these.
fn is_commit_hash(revision: &str) -> bool {
    revision.len() == 40 && revision.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RepoType {
    #[default]
    Model,
    Dataset,
    Space,
}

impl RepoType {
    /// The cache directory prefix (`models--org--name` etc.).
    fn cache_prefix(self) -> &'static str {
        match self {
            RepoType::Model => "models",
            RepoType::Dataset => "datasets",
            RepoType::Space => "spaces",
        }
    }

    /// The URL path prefix before the repo id.
    fn url_prefix(self) -> &'static str {
        match self {
            RepoType::Model => "",
            RepoType::Dataset => "datasets/",
            RepoType::Space => "spaces/",
        }
    }
}

impl std::str::FromStr for RepoType {
    type Err = eyre::Report;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "model" => Ok(RepoType::Model),
            "dataset" => Ok(RepoType::Dataset),
            "space" => Ok(RepoType::Space),
            _ => Err(eyre::eyre!("unknown repo_type {s:?}: expected \"model\", \"dataset\", or \"space\"")),
        }
    }
}

/// The cache directory of a repo (`models--org--name` etc.).
fn repo_cache_dir(repo_type: RepoType, repo_id: &str) -> PathBuf {
    hf_hub_cache_dir().join(format!("{}--{}", repo_type.cache_prefix(), repo_id.replace('/', "--")))
}

/// Model-repo [`cached_hub_file_in`].
pub fn cached_hub_file(repo_id: &str, filename: &str, revision: &str) -> Option<PathBuf> {
    cached_hub_file_in(RepoType::Model, repo_id, filename, revision)
}

/// Path of `filename` in the local HF cache, or None when not cached.
///
/// A pure-filesystem lookup — no request is made. `revision` may be a commit
/// hash (used directly as the snapshot name) or a branch/tag name (followed
/// through the cached ref).
pub fn cached_hub_file_in(repo_type: RepoType, repo_id: &str, filename: &str, revision: &str) -> Option<PathBuf> {
    let repo_dir = repo_cache_dir(repo_type, repo_id);
    let commit_owned;
    let commit = if is_commit_hash(revision) {
        revision
    } else {
        commit_owned = std::fs::read_to_string(repo_dir.join("refs").join(revision)).ok()?;
        commit_owned.trim()
    };
    let path = repo_dir.join("snapshots").join(commit).join(filename);
    path.is_file().then_some(path)
}

/// Download failure with a definite HTTP cause, kept as a typed error so the
/// Python bindings can raise the matching exception (FileNotFoundError for
/// 404, PermissionError for 401/403).
#[derive(Debug)]
pub enum FetchError {
    /// 404: no such repo, revision, or file.
    NotFound { url: String },
    /// 401/403: private or gated repo.
    Unauthorized { url: String, status: u16, had_token: bool },
}

impl fmt::Display for FetchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FetchError::NotFound { url } => {
                write!(f, "{url}: HTTP 404 — no such repo with that file, and no such local file either")
            }
            FetchError::Unauthorized { url, status, had_token } => {
                let token = if *had_token {
                    "the request used the discovered token"
                } else {
                    "no token was found"
                };
                write!(
                    f,
                    "{url}: HTTP {status} — the repo may be private or gated ({token}; set HF_TOKEN or run \
                     `hf auth login`, and accept the repo's terms on huggingface.co if it is gated)"
                )
            }
        }
    }
}

impl std::error::Error for FetchError {}

/// Model-repo [`hub_file_in`].
pub fn hub_file(repo_id: &str, filename: &str, revision: &str) -> Result<PathBuf> {
    hub_file_in(RepoType::Model, repo_id, filename, revision)
}

/// Path of `filename` from Hub repo `repo_id` at `revision`, served from the
/// standard HF cache, downloading into it first when absent.
pub fn hub_file_in(repo_type: RepoType, repo_id: &str, filename: &str, revision: &str) -> Result<PathBuf> {
    if let Some(path) = cached_hub_file_in(repo_type, repo_id, filename, revision) {
        return Ok(path);
    }
    download_into_cache(repo_type, repo_id, filename, revision)
        .wrap_err_with(|| format!("downloading {filename} from Hub repo {repo_id} at revision {revision}"))
}

/// GET `endpoint/repo/resolve/revision/filename` and stream the body into the
/// cache snapshot named by the `x-repo-commit` response header, recording the
/// branch ref so later lookups (ours and huggingface_hub's) resolve it.
fn download_into_cache(repo_type: RepoType, repo_id: &str, filename: &str, revision: &str) -> Result<PathBuf> {
    let endpoint = env_nonempty("HF_ENDPOINT").unwrap_or_else(|| "https://huggingface.co".to_owned());
    let url = format!(
        "{}/{}{repo_id}/resolve/{revision}/{filename}",
        endpoint.trim_end_matches('/'),
        repo_type.url_prefix()
    );
    let token = get_hf_token();

    // Redirects are followed by hand: resolve/ URLs answer with the
    // `x-repo-commit` header and a redirect to a CDN for LFS files, and the
    // Authorization header must not travel to the other host.
    let agent = ureq::Agent::config_builder()
        .max_redirects(0)
        .http_status_as_error(false)
        .build()
        .new_agent();
    let mut request = agent.get(&url).header("User-Agent", "gigatoken");
    if let Some(token) = &token {
        request = request.header("Authorization", format!("Bearer {token}"));
    }
    let mut response = request.call().wrap_err_with(|| format!("requesting {url}"))?;
    let header = |resp: &ureq::http::Response<ureq::Body>, name: &str| {
        resp.headers().get(name).and_then(|v| v.to_str().ok()).map(str::to_owned)
    };
    let commit = header(&response, "x-repo-commit");

    let mut hops = 0;
    while response.status().is_redirection() {
        hops += 1;
        ensure_status(&url, response.status().as_u16(), token.is_some())?;
        if hops > 10 {
            bail!("{url}: too many redirects");
        }
        let location = header(&response, "location")
            .ok_or_else(|| eyre::eyre!("{url}: redirect with no Location header"))?;
        let next_url = absolutize(&location, &url);
        // No Authorization here: the redirect target is a presigned CDN URL
        // on another host (requests/huggingface_hub drop the header too).
        response = agent
            .get(&next_url)
            .header("User-Agent", "gigatoken")
            .call()
            .wrap_err_with(|| format!("requesting {next_url}"))?;
    }
    ensure_status(&url, response.status().as_u16(), token.is_some())?;

    // Snapshot directory: the commit the Hub reported, falling back to the
    // requested revision (e.g. a plain file server behind HF_ENDPOINT).
    let commit = commit.unwrap_or_else(|| revision.to_owned());
    let repo_dir = repo_cache_dir(repo_type, repo_id);
    let target = repo_dir.join("snapshots").join(&commit).join(filename);
    let dir = target.parent().expect("snapshot file has a parent");
    std::fs::create_dir_all(dir).wrap_err_with(|| format!("creating {}", dir.display()))?;

    // Stream to a sibling temp file, then rename: concurrent downloaders
    // race benignly and readers never observe a partial file.
    let tmp = dir.join(format!(".{}.{}.tmp", target.file_name().unwrap().to_string_lossy(), std::process::id()));
    let result = (|| -> Result<()> {
        let mut file = std::fs::File::create(&tmp)?;
        io::copy(&mut response.body_mut().as_reader(), &mut file)?;
        file.sync_all()?;
        std::fs::rename(&tmp, &target)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result.wrap_err_with(|| format!("writing {}", target.display()))?;

    if !is_commit_hash(revision) && revision != commit {
        let refs_dir = repo_dir.join("refs");
        std::fs::create_dir_all(&refs_dir)
            .and_then(|_| std::fs::write(refs_dir.join(revision), &commit))
            .wrap_err_with(|| format!("recording ref {revision} -> {commit}"))?;
    }
    Ok(target)
}

fn ensure_status(url: &str, status: u16, had_token: bool) -> Result<()> {
    match status {
        200..=399 => Ok(()),
        404 => Err(FetchError::NotFound { url: url.to_owned() }.into()),
        401 | 403 => Err(FetchError::Unauthorized { url: url.to_owned(), status, had_token }.into()),
        _ => Err(eyre::eyre!("{url}: HTTP {status}")),
    }
}

/// A redirect Location resolved against the request URL: absolute URLs pass
/// through, host-relative (`/x/y`) and path-relative ones join the base.
fn absolutize(location: &str, base: &str) -> String {
    if location.contains("://") {
        return location.to_owned();
    }
    let origin_len = base.find("://").map(|i| i + 3).unwrap_or(0);
    let origin_end = base[origin_len..].find('/').map_or(base.len(), |i| origin_len + i);
    if location.starts_with('/') {
        format!("{}{location}", &base[..origin_end])
    } else {
        let dir_end = base.rfind('/').unwrap_or(base.len());
        format!("{}/{location}", &base[..dir_end.max(origin_end)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_id_shapes() {
        assert!(looks_like_repo_id("gpt2"));
        assert!(looks_like_repo_id("openai-community/gpt2"));
        assert!(looks_like_repo_id("Qwen/Qwen3.5-9B"));
        assert!(!looks_like_repo_id("data/tokenizers/gpt2.json"));
        assert!(!looks_like_repo_id("./gpt2"));
        assert!(!looks_like_repo_id("/abs/path"));
        assert!(!looks_like_repo_id("gpt2_tokenizer.json"));
        assert!(!looks_like_repo_id("subdir/tokenizer.model"));
        assert!(!looks_like_repo_id(""));
        assert!(!looks_like_repo_id("org/"));
    }

    #[test]
    fn location_resolution() {
        assert_eq!(absolutize("https://cdn.example/x", "https://huggingface.co/a/b"), "https://cdn.example/x");
        assert_eq!(absolutize("/api/x", "https://huggingface.co/a/b"), "https://huggingface.co/api/x");
        assert_eq!(absolutize("y", "https://huggingface.co/a/b"), "https://huggingface.co/a/y");
    }
}
