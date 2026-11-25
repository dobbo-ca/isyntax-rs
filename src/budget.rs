use crate::Error;
use crate::error::check_cancel;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, MutexGuard};

/// Locks a mutex whose holder panicked: the data is plain counters and caches, and a panic is
/// already reported as an error by the worker that caught it.
pub(crate) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The memory limit of an image and its views: caches and scratch charge it before they allocate.
/// Go keeps one per process (`budget`).
pub(crate) struct Budget {
    state: Mutex<(u64, u64, u64)>, // (limit, used, peak of used)
}

impl Budget {
    pub(crate) fn new(limit: u64) -> Budget {
        Budget {
            state: Mutex::new((limit, 0, 0)),
        }
    }

    /// What is left of the limit.
    #[cfg(test)]
    pub(crate) fn free(&self) -> u64 {
        let s = lock(&self.state);
        s.0.saturating_sub(s.1)
    }

    /// The most bytes held at any one time.
    #[cfg(test)]
    pub(crate) fn peak(&self) -> u64 {
        lock(&self.state).2
    }

    /// Sets the limit and returns the previous one.
    pub(crate) fn set_limit(&self, n: u64) -> u64 {
        std::mem::replace(&mut lock(&self.state).0, n)
    }

    pub(crate) fn limit(&self) -> u64 {
        lock(&self.state).0
    }

    /// Charges `n` bytes: [`Error::AllocLimit`] if `n` alone is over the limit,
    /// [`Error::MemoryLimit`] if what is held makes it so.
    pub(crate) fn reserve(&self, n: u64) -> Result<(), Error> {
        let mut s = lock(&self.state);
        if n > s.0 {
            return Err(Error::AllocLimit);
        }
        if s.1 > s.0 - n {
            return Err(Error::MemoryLimit);
        }
        s.1 += n;
        s.2 = s.2.max(s.1);
        Ok(())
    }

    pub(crate) fn release(&self, n: u64) {
        let mut s = lock(&self.state);
        s.1 = s.1.saturating_sub(n);
    }
}

/// Bytes charged to a [`Budget`], released when dropped.
pub(crate) struct Charge {
    budget: Arc<Budget>,
    n: u64,
}

impl Charge {
    pub(crate) fn new(budget: &Arc<Budget>, n: u64) -> Result<Charge, Error> {
        budget.reserve(n)?;
        Ok(Charge {
            budget: budget.clone(),
            n,
        })
    }
}

impl Drop for Charge {
    fn drop(&mut self) {
        self.budget.release(self.n);
    }
}

/// Running total of what the open of an image (the parse of its header, then the tile index)
/// allocates, against `Limits::max_alloc`; it also counts the units of work of the parse, to check the cancel flag between them.
pub(crate) struct Acct<'a> {
    used: u64,
    max: u64,
    cancel: Option<&'a AtomicBool>,
    ticks: u32,
}

impl<'a> Acct<'a> {
    pub(crate) fn new(max: u64) -> Acct<'a> {
        Acct {
            used: 0,
            max,
            cancel: None,
            ticks: 0,
        }
    }

    pub(crate) fn with_cancel(max: u64, cancel: &'a AtomicBool) -> Acct<'a> {
        Acct {
            cancel: Some(cancel),
            ..Acct::new(max)
        }
    }

    /// What has been charged so far.
    #[cfg(test)]
    pub(crate) fn used(&self) -> u64 {
        self.used
    }

    pub(crate) fn add(&mut self, n: u64) -> Result<(), Error> {
        self.used = self.used.saturating_add(n);
        if self.used > self.max {
            return Err(Error::AllocLimit);
        }
        Ok(())
    }

    /// One element, table entry or tile done: every 4096th checks the cancel flag.
    pub(crate) fn tick(&mut self) -> Result<(), Error> {
        self.ticks = self.ticks.wrapping_add(1);
        match self.cancel {
            Some(c) if self.ticks.is_multiple_of(4096) => check_cancel(c),
            _ => Ok(()),
        }
    }
}
