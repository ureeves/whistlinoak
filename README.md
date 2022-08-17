# Whistlin'Oak

[![Repository](https://img.shields.io/badge/github-whistlinoak-darkgreen?logo=github)](https://github.com/ureeves/whistlinoak)
![Build Status](https://github.com/ureeves/whistlinoak/workflows/build/badge.svg)
[![Documentation](https://img.shields.io/badge/docs-whistlinoak-orange?logo=rust)](https://docs.rs/whistlinoak/)

Annotated even-arity trees backed by arbitrary memories.

## Usage
```toml
whistlinoak = "0.1"
```

## Example
```rust 
use whistlinoak::{Annotation, Tree};

struct Cardinality(usize);

impl<T> Annotation<T> for Cardinality {
    fn compute(leaf: &T) -> Self {
        Cardinality(1)
    }

    fn combine<'a>(annotations: impl Iterator<Item=&'a Self>) -> Self
    where
        Self: 'a
    {
        Self(annotations.fold(0, |curr, c| curr + c.0))
    }
}

let mut tree = Tree::<usize, Cardinality>::new();

let n_leaves = 1000;

for i in 0..n_leaves {
    tree.push(i).unwrap();
}

assert_eq!(tree.len(), n_leaves);
assert_eq!(tree.root().0, n_leaves);
```
