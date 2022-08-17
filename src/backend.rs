//! Pluggable memory backends for a [`Tree`].
//!
//! [`Tree`]: `crate::Tree`

use std::convert::Infallible;
use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;

use memmap2::MmapMut;

/// A type that manages the memory available to a tree.
pub trait Memory {
    /// The error a backend can return.
    type Error;

    /// Return a read-only slice of the entire  memory managed by this
    /// backend.
    fn memory(&self) -> &[u8];
    /// Return a writeable slice of the entire memory managed by this
    /// backend.
    fn memory_mut(&mut self) -> &mut [u8];
    /// Set the size of the memory managed by this backend.
    fn set_len(&mut self, size: usize) -> Result<(), Self::Error>;
}

/// A [`Memory`] implementation backed by a file using an mmap.
pub struct FileMemory {
    file: File,
    mmap: MmapMut,
}

impl FileMemory {
    pub(crate) fn new<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(path)?;

        let mmap = unsafe { MmapMut::map_mut(&file)? };
        Ok(Self { file, mmap })
    }
}

impl Memory for FileMemory {
    type Error = io::Error;

    fn memory(&self) -> &[u8] {
        self.mmap.as_ref()
    }

    fn memory_mut(&mut self) -> &mut [u8] {
        self.mmap.as_mut()
    }

    fn set_len(&mut self, size: usize) -> Result<(), Self::Error> {
        self.file.set_len(size as u64)?;
        self.mmap = unsafe { MmapMut::map_mut(&self.file)? };
        Ok(())
    }
}

impl Memory for Vec<u8> {
    type Error = Infallible;

    fn memory(&self) -> &[u8] {
        self
    }

    fn memory_mut(&mut self) -> &mut [u8] {
        self
    }

    fn set_len(&mut self, size: usize) -> Result<(), Self::Error> {
        self.resize(size, 0);
        Ok(())
    }
}
