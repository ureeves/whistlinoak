#![doc = include_str!("../README.md")]
#![deny(missing_docs)]
#![deny(clippy::all)]

use std::fs::{File, OpenOptions};
use std::io::{self, Error, ErrorKind};
use std::marker::PhantomData;
use std::mem::{self, MaybeUninit};
use std::path::Path;
use std::ptr;

use memmap2::MmapMut;

/// Annotation over a leaf.
pub trait Annotation<T> {
    /// Return the annotation for the given leaf.
    fn compute(leaf: &T) -> Self;

    /// Combine multiple annotations into another annotation.
    fn combine<'a>(annotations: impl Iterator<Item = &'a Self>) -> Self
    where
        Self: 'a;
}

/// The unit struct annotates everything
impl<T> Annotation<T> for () {
    fn compute(_: &T) -> Self {}

    fn combine<'a>(_: impl Iterator<Item = &'a Self>) -> Self
    where
        Self: 'a,
    {
    }
}

/// Kept at the beginning of the mmap to speed up operations.
struct Header {
    filled: usize,
    cap: usize,
    levels: usize,
}

/// An annotated tree structure with an arity of `2N`, backed by an mmap.
///
/// Each leaf contained in the tree is decorated by an [`Annotation`], computed
/// on the [`push`] of said leaf. The change is then propagated up through the
/// tree, recursively combining the annotations of filled subtrees until the
/// [`root`] itself is computed.
///
/// Leaves can also be [`pop`]ped from tree, and annotations will be kept
/// updated.
///
/// # Safety
/// Using an `N` of 0 will result in a panic when trying to perform any
/// operation.
///
/// [`push`]: Tree::push
/// [`pop`]: Tree::pop
/// [`root`]: Tree::root
pub struct Tree<L, A = (), const N: usize = 1> {
    mmap: MmapMut,
    file: File,

    marker_t: PhantomData<*const L>,
    marker_a: PhantomData<*const A>,
}

impl<L, A, const N: usize> Tree<L, A, N>
where
    A: Annotation<L>,
{
    /// Creates a new tree, mmapped to the file at the given path.
    ///
    /// # Safety
    /// When a file of length smaller than the tree header (three [`usize`]s)
    /// exists at the given path, it will be assumed to contain a valid tree. If
    /// not, a new file will be created.
    pub fn new<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(path)?;
        let meta = file.metadata()?;

        let unit_size = unit_tree_size::<L, A, N>() as u64;

        let file_len = meta.len();
        let header_size = header_size() as u64;

        // if the file is smaller than metadata length, grow the file and
        // write an empty tree header.
        if file_len < header_size {
            file.set_len(header_size + unit_size)?;
            unsafe {
                let mut mmap = MmapMut::map_mut(&file)?;

                let header_ptr = mmap.as_mut_ptr();
                let header_ptr = header_ptr as *mut Header;

                ptr::write(
                    header_ptr,
                    Header {
                        filled: 0,
                        cap: arity(N),
                        levels: 1,
                    },
                )
            };
        }

        let mmap = unsafe { MmapMut::map_mut(&file)? };

        let mut tree = Self {
            mmap,
            file,
            marker_a: Default::default(),
            marker_t: Default::default(),
        };

        let (tree_buf, header) = tree.mut_tree_buf_and_header();

        // if the tree is not the expected size error out since it is
        // undoubtedly corrupt
        if tree_buf.len() != tree_size::<L, A, N>(header.levels) {
            return Err(Error::new(ErrorKind::InvalidData, "corrupt tree"));
        }

        Ok(tree)
    }

    /// Push a leaf to the tree.
    pub fn push(&mut self, leaf: L) -> io::Result<()> {
        let (_, header) = self.tree_buf_and_header();

        if header.filled == header.cap {
            self.grow_tree()?;
        }

        let (tree, header) = self.mut_tree_buf_and_header();

        unsafe {
            let view = TreeViewMut::<L, A, N>::new(
                tree.as_mut_ptr(),
                header.filled,
                tree.len(),
                header.cap,
            );
            view.traverse(A::combine, |left_leaves, right_leaves, filled| {
                match filled < N {
                    true => {
                        left_leaves[filled].1 = A::compute(&leaf);
                        left_leaves[filled].0 = leaf;
                    }
                    false => {
                        let index = filled - N;
                        right_leaves[index].1 = A::compute(&leaf);
                        right_leaves[index].0 = leaf;
                    }
                };

                A::combine(leaf_iter(left_leaves, right_leaves, filled + 1))
            });
        }

        header.filled += 1;

        Ok(())
    }

    /// Pop the last leaf from the tree.
    pub fn pop(&mut self) -> io::Result<Option<L>> {
        let (_, header) = self.tree_buf_and_header();

        if header.filled == 0 {
            return Ok(None);
        }

        let (tree, header) = self.mut_tree_buf_and_header();

        let popped = unsafe {
            let mut popped: MaybeUninit<L> = MaybeUninit::uninit();

            let view = TreeViewMut::<L, A, N>::new(
                tree.as_mut_ptr(),
                header.filled - 1,
                tree.len(),
                header.cap,
            );
            view.traverse(A::combine, |left_leaves, right_leaves, filled| {
                match filled < N {
                    true => ptr::copy(
                        &left_leaves[filled].0,
                        popped.as_mut_ptr(),
                        1,
                    ),
                    false => ptr::copy(
                        &right_leaves[filled - N].0,
                        popped.as_mut_ptr(),
                        1,
                    ),
                };

                A::combine(leaf_iter(left_leaves, right_leaves, filled))
            });

            popped.assume_init()
        };

        header.filled -= 1;

        let shrunk_cap = header.cap / arity(N);

        // don't shrink if the capacity is as low as it gets
        if header.filled == shrunk_cap && header.cap != arity(N) {
            self.shrink_tree()?;
        }

        Ok(Some(popped))
    }

    /// Returns a reference to the root annotation
    pub fn root(&self) -> &A {
        let root_offset = self.root_offset();
        let (tree, _) = self.tree_buf_and_header();

        let ptr = tree[root_offset..].as_ptr();
        let ptr = ptr as *const A;

        unsafe { &*ptr }
    }

    /// Return the number of leaves in the tree
    pub fn len(&self) -> usize {
        let (_, header) = self.tree_buf_and_header();
        header.filled
    }

    /// Return the current capacity of the tree.
    pub fn cap(&self) -> usize {
        let (_, header) = self.tree_buf_and_header();
        header.cap
    }

    /// Return true if the tree is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Double the growth of the tree and reserve space in the middle for an
    /// annotation
    fn grow_tree(&mut self) -> io::Result<()> {
        let (_, header) = self.tree_buf_and_header();

        let len = header_size() + tree_size::<L, A, N>(header.levels + 1);
        let len = len as u64;

        self.file.set_len(len)?;

        let (_, header) = self.mut_tree_buf_and_header();

        header.cap *= arity(N);
        header.levels += 1;

        self.mmap = unsafe { MmapMut::map_mut(&self.file)? };

        Ok(())
    }

    fn shrink_tree(&mut self) -> io::Result<()> {
        let (_, header) = self.tree_buf_and_header();

        let len = header_size() + tree_size::<L, A, N>(header.levels - 1);
        let len = len as u64;

        self.mmap = MmapMut::map_anon(0)?;
        self.file.set_len(len)?;

        let (_, header) = self.mut_tree_buf_and_header();

        header.cap /= arity(N);
        header.levels -= 1;

        self.mmap = unsafe { MmapMut::map_mut(&self.file)? };

        Ok(())
    }

    /// Returns the offset of the root annotation in the tree buffer
    fn root_offset(&self) -> usize {
        let (tree, _) = self.tree_buf_and_header();
        (tree.len() - mem::size_of::<A>()) / 2
    }

    fn tree_buf_and_header(&self) -> (&[u8], &Header) {
        let (header, tree) = self.mmap.split_at(header_size());

        let ptr = header.as_ptr();
        let ptr = ptr as *const Header;

        unsafe { (tree, &*ptr) }
    }

    fn mut_tree_buf_and_header(&mut self) -> (&mut [u8], &mut Header) {
        let (header, tree) = self.mmap.split_at_mut(header_size());

        let ptr = header.as_mut_ptr();
        let ptr = ptr as *mut Header;

        unsafe { (tree, &mut *ptr) }
    }
}

struct TreeViewMut<'a, T, A, const N: usize> {
    tree: *mut u8,
    filled: usize,
    tree_len: usize,
    tree_cap: usize,

    marker_t: PhantomData<*const T>,
    marker_a: PhantomData<*const A>,
    marker_lifetime: PhantomData<&'a A>,
}

impl<'a, T, A, const N: usize> TreeViewMut<'a, T, A, N>
where
    A: Annotation<T>,
{
    fn new(
        tree: *mut u8,
        filled: usize,
        tree_len: usize,
        tree_cap: usize,
    ) -> Self {
        Self {
            tree,
            filled,
            tree_len,
            tree_cap,
            marker_a: Default::default(),
            marker_t: Default::default(),
            marker_lifetime: Default::default(),
        }
    }

    unsafe fn traverse<FA, FL>(&self, anno_closure: FA, leaves_closure: FL)
    where
        FL: FnOnce(&mut [(T, A); N], &mut [(T, A); N], usize) -> A,
        FA: Copy + Fn(AnnoIter<'a, A, N>) -> A,
    {
        let anno_size = mem::size_of::<A>();
        let anno_offset = (self.tree_len - anno_size) / 2;

        let anno = self.tree.add(anno_offset) as *mut A;

        match self.tree_len == unit_tree_size::<T, A, N>() {
            true => {
                let left = self.tree as *mut [(T, A); N];
                let right =
                    self.tree.add(anno_offset + anno_size) as *mut [(T, A); N];

                *anno = leaves_closure(&mut *left, &mut *right, self.filled);
            }
            false => {
                let subtree_len = (self.tree_len - anno_size) / arity(N);
                let subtree_cap = self.tree_cap / arity(N);

                let subtree_index = self.filled / subtree_cap;
                let subtree = match subtree_index < N {
                    true => {
                        let subtree_offset = subtree_index * subtree_len;
                        self.tree.add(subtree_offset)
                    }
                    false => {
                        let subtree_offset = anno_offset
                            + anno_size
                            + (subtree_index - N) * subtree_len;
                        self.tree.add(subtree_offset)
                    }
                };

                let subtree_filled = self.filled - subtree_index * subtree_cap;
                let view = TreeViewMut::<T, A, N>::new(
                    subtree,
                    subtree_filled,
                    subtree_len,
                    subtree_cap,
                );

                view.traverse(anno_closure, leaves_closure);

                *anno = anno_closure(AnnoIter::new(
                    self.tree,
                    subtree_len,
                    subtree_index,
                ));
            }
        }
    }
}

struct AnnoIter<'a, A, const N: usize> {
    tree: *const u8,
    subtree_len: usize,
    subtree_index: usize,
    index: usize,

    marker_a: PhantomData<*const A>,
    market_lifetime: PhantomData<&'a A>,
}

impl<'a, A, const N: usize> AnnoIter<'a, A, N> {
    fn new(tree: *const u8, subtree_len: usize, subtree_index: usize) -> Self {
        Self {
            tree,
            subtree_len,
            subtree_index,
            index: 0,
            marker_a: Default::default(),
            market_lifetime: Default::default(),
        }
    }
}

impl<'a, A, const N: usize> Iterator for AnnoIter<'a, A, N> {
    type Item = &'a A;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            if self.index > self.subtree_index {
                return None;
            }

            let anno_size = mem::size_of::<A>();
            let offset = (self.subtree_len - anno_size) / 2;
            let offset = offset + self.index * self.subtree_len;

            let anno = match self.index < N {
                true => self.tree.add(offset),
                false => self.tree.add(offset + anno_size),
            } as *const A;

            self.index += 1;

            Some(&*anno)
        }
    }
}

const fn header_size() -> usize {
    mem::size_of::<Header>()
}

const fn arity(n: usize) -> usize {
    2 * n
}

const fn tree_size<L, A, const N: usize>(levels: usize) -> usize {
    let anno_size = mem::size_of::<A>();
    let leaf_size = mem::size_of::<L>();

    let arity = arity(N);

    let mut size = arity * (leaf_size + anno_size) + anno_size;
    let mut i = 1;

    while i < levels {
        size = arity * size + anno_size;
        i += 1;
    }

    size
}

const fn unit_tree_size<L, A, const N: usize>() -> usize {
    tree_size::<L, A, N>(1)
}

fn leaf_iter<'a, T, A, const N: usize>(
    left_leaves: &'a [(T, A); N],
    right_leaves: &'a [(T, A); N],
    filled: usize,
) -> impl Iterator<Item = &'a A>
where
    A: Annotation<T>,
{
    left_leaves
        .iter()
        .chain(right_leaves.iter())
        .enumerate()
        .filter(move |(i, _)| *i < filled)
        .map(|(_, (_, a))| a)
}

#[cfg(test)]
mod tests {
    use crate::{Annotation, Tree};

    use tempfile::NamedTempFile;

    const N_ELEMS: usize = 2048;

    struct Cardinality(usize);

    impl<T> Annotation<T> for Cardinality {
        fn compute(_: &T) -> Self {
            Cardinality(1)
        }

        fn combine<'a>(annotations: impl Iterator<Item = &'a Self>) -> Self
        where
            Self: 'a,
        {
            Self(annotations.fold(0, |curr, c| curr + c.0))
        }
    }

    fn cardinality<const N: usize>() {
        let file = NamedTempFile::new().expect("there should be a tmp file");
        let path = file.into_temp_path();

        let mut tree = Tree::<usize, Cardinality, N>::new(path)
            .expect("should open the tree successfully");

        for i in 0..N_ELEMS {
            tree.push(i).expect("should push to the tree successfully");
        }

        assert_eq!(tree.len(), tree.root().0);

        for _ in 0..N_ELEMS {
            tree.pop().expect("should pop from the tree successfully");
        }

        assert_eq!(tree.len(), tree.root().0);
    }

    fn insert<const N: usize>() {
        let file = NamedTempFile::new().expect("there should be a tmp file");
        let path = file.into_temp_path();

        let mut tree = Tree::<usize, (), N>::new(path)
            .expect("creating a tree should go ok");

        for i in 0..N_ELEMS {
            tree.push(i).expect("push should succeed");
        }

        assert_eq!(tree.len(), N_ELEMS as usize);
    }

    fn insert_and_pop<const N: usize>() {
        let file = NamedTempFile::new().expect("there should be a tmp file");
        let path = file.into_temp_path();

        let mut tree = Tree::<usize, (), N>::new(path)
            .expect("creating a tree should go ok");

        for i in 0..N_ELEMS {
            tree.push(i).expect("push should succeed");
        }

        for i in (0..N_ELEMS).rev() {
            assert_eq!(
                i,
                tree.pop()
                    .expect("pop should succeed")
                    .expect("pop should yield a leaf")
            );
        }
        assert!(tree.is_empty());
    }

    macro_rules! tests {
        ( $( $n:literal ),* ) => {
            #[test]
            fn inserts_and_pops() {
                $(
                    insert_and_pop::<$n>();
                )*
            }

            #[test]
            fn inserts() {
                $(
                    insert::<$n>();
                )*
            }

            #[test]
            fn cardinalities() {
                $(
                    cardinality::<$n>();
                )*
            }
        };
    }

    tests!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
}
