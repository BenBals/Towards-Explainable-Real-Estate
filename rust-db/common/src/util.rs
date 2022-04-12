//! A module which contains some utility functions
use serde::Serialize;
use std::{
    env,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

use super::BpResult;

/// Write something which can be serialized to a json file at the specified path.
/// Returns an Error if the file can't be written or the directory can't be created.
pub fn write_serializable_to_json<P: AsRef<Path>>(
    output: &impl Serialize,
    path: P,
) -> BpResult<()> {
    let json_string = serde_json::to_string(output).expect("Serialization of Output can't fail");

    if let Some(parent) = path.as_ref().parent() {
        fs::create_dir_all(parent)?;
    }

    let mut file_handle = File::create(path)?;
    file_handle.write_all(json_string.as_bytes())?;

    Ok(())
}

/// Finds the project root, that is the root of the git repo.
/// In particular, this will return the path to the closest ancestor to the current working directory,
/// which contains a `.git` folder.
/// If no such ancestor is found, the current working directory is retuned.
pub fn find_project_root() -> BpResult<PathBuf> {
    let cwd = env::current_dir()?;

    #[allow(clippy::redundant_closure)]
    Ok(cwd
        .ancestors()
        .find(|ancestor| has_git_directory(ancestor))
        .map(PathBuf::from)
        .unwrap_or_else(|| cwd))
}

fn has_git_directory<P>(path: P) -> bool
where
    P: AsRef<Path>,
{
    let mut path_buf = path.as_ref().to_path_buf();
    path_buf.push(".git");
    // This also checks if the path exists.
    path_buf.is_dir()
}

/// Gives you either the given path or a your specified relative path on the project root
/// If `path = Some(path_buf)` returns cloned `path_buf`, else `project_root/{relative_path}`
/// If the folder does not exist yet, it will be created.
pub fn path_or_relative_to_project_root(path: Option<&PathBuf>, relative_path: &str) -> PathBuf {
    path.cloned().unwrap_or_else(|| {
        let mut result = find_project_root().unwrap();
        result.push(relative_path);

        if let Some(path) = result.parent() {
            fs::create_dir_all(path).unwrap();
        }
        result
    })
}
