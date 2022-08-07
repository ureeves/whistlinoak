use blake2b_simd::{Hash as Blake2bHash, Params, State};
use clap::Parser;
use std::path::{Path, PathBuf};
use tempfile::NamedTempFile;
use whistlinoak::{Annotation, Tree};

// the hash of a leaf
struct Hash(Blake2bHash);

fn hash_state() -> State {
    Params::new().hash_length(16).to_state()
}

impl Annotation<u16> for Hash {
    // compute the hash from the leaf
    fn compute(leaf: &u16) -> Self {
        let mut state = hash_state();
        state.update(&leaf.to_le_bytes());
        Self(state.finalize())
    }

    // combine different hashes
    fn combine<'a>(annotations: impl Iterator<Item = &'a Self>) -> Self
    where
        Self: 'a,
    {
        let mut state = hash_state();
        for h in annotations {
            state.update(h.0.as_bytes());
        }
        Self(state.finalize())
    }
}

#[derive(Parser)]
struct Args {
    file: Option<PathBuf>,
}

fn main() {
    let args = Args::parse();

    let path: Box<dyn AsRef<Path>> = match args.file {
        Some(path) => Box::new(path),
        None => {
            let file = NamedTempFile::new()
                .expect("creating a temp file shouldn't fail");
            Box::new(file.into_temp_path())
        }
    };

    let mut tree = Tree::<u16, Hash>::new(path.as_ref())
        .expect("creating new tree to be successful");

    let mut state = hash_state();

    // compute the hash of the left subtree manually
    let mut left_state = hash_state();
    for i in 0..2 {
        let mut inner_state = hash_state();

        tree.push(i).expect("push should be successful");
        inner_state.update(&i.to_le_bytes());

        left_state.update(inner_state.finalize().as_bytes());
    }
    state.update(left_state.finalize().as_bytes());

    // compute the hash of the right subtree
    let mut right_state = hash_state();
    for i in 2..4 {
        let mut inner_state = hash_state();

        tree.push(i).expect("push should be successful");
        inner_state.update(&i.to_le_bytes());

        right_state.update(inner_state.finalize().as_bytes());
    }
    state.update(right_state.finalize().as_bytes());

    let root = state.finalize();

    println!("These hashes | {}", root.to_hex());
    println!("should match | {}", tree.root().0.to_hex());
}
