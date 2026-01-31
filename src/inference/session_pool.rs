//! Session pool utilities for concurrent model inference.
//!
//! Provides [`SessionPool`] and [`KeyedSessionPool`] for managing multiple
//! model instances that can serve inference requests concurrently. Uses ort's
//! [`PrepackedWeights`] to share model weights across pooled sessions,
//! reducing memory overhead.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::OnceLock;

use ort::session::builder::PrepackedWeights;
use parking_lot::{Mutex, MutexGuard, RwLock};

use crate::inference::InferenceError;

/// Fallible init for `OnceLock<T>`: returns existing value or attempts to
/// initialize, returning an error if init fails without poisoning the lock.
pub fn once_lock_try_init<T, F>(lock: &OnceLock<T>, init: F) -> Result<&T, InferenceError>
where
    F: FnOnce() -> Result<T, InferenceError>,
{
    if let Some(val) = lock.get() {
        return Ok(val);
    }
    let val = init()?;
    let _ = lock.set(val);
    lock.get().ok_or_else(|| InferenceError::ProcessingError {
        message: "Failed to initialize pool".to_string(),
    })
}

/// Acquires a lock from the pool using try_lock round-robin,
/// falling back to blocking on slot 0 if all are contended.
fn acquire<T>(pool: &[Mutex<T>]) -> MutexGuard<'_, T> {
    for m in pool.iter() {
        if let Some(guard) = m.try_lock() {
            return guard;
        }
    }
    pool[0].lock()
}

/// A pool of N identical model instances sharing [`PrepackedWeights`].
///
/// Each instance is behind its own [`Mutex`], allowing up to N concurrent
/// inference calls. The pool distributes access via try_lock round-robin.
pub struct SessionPool<T> {
    instances: Vec<Mutex<T>>,
}

impl<T> SessionPool<T> {
    /// Creates a pool of `pool_size` instances, sharing a single
    /// [`PrepackedWeights`] across all sessions for memory efficiency.
    pub fn new<F>(pool_size: usize, init: F) -> Result<Self, InferenceError>
    where
        F: Fn(&PrepackedWeights) -> Result<T, InferenceError>,
    {
        let weights = PrepackedWeights::new();
        let mut instances = Vec::with_capacity(pool_size);
        for _ in 0..pool_size {
            instances.push(Mutex::new(init(&weights)?));
        }
        Ok(Self { instances })
    }

    /// Executes a closure with exclusive access to one pooled instance.
    pub fn with<F, R>(&self, f: F) -> Result<R, InferenceError>
    where
        F: FnOnce(&mut T) -> Result<R, InferenceError>,
    {
        let mut guard = acquire(&self.instances);
        f(&mut guard)
    }
}

/// A keyed pool for models with dynamic keys (e.g., language-specific CRNN).
///
/// Each key maps to its own [`SessionPool`], allowing concurrent access
/// both across keys and within the same key's pool.
pub struct KeyedSessionPool<K, T> {
    map: RwLock<HashMap<K, SessionPool<T>>>,
}

impl<K: Eq + Hash + Clone, T> KeyedSessionPool<K, T> {
    /// Creates an empty keyed pool.
    pub fn new() -> Self {
        Self {
            map: RwLock::new(HashMap::new()),
        }
    }

    /// Ensures a pool exists for the given key, creating one if needed.
    pub fn get_or_init<F>(&self, key: K, pool_size: usize, init: F) -> Result<(), InferenceError>
    where
        F: Fn(&PrepackedWeights) -> Result<T, InferenceError>,
    {
        {
            if self.map.read().contains_key(&key) {
                return Ok(());
            }
        }
        let mut write = self.map.write();
        if let std::collections::hash_map::Entry::Vacant(entry) = write.entry(key) {
            let pool = SessionPool::new(pool_size, init)?;
            entry.insert(pool);
        }
        Ok(())
    }

    /// Executes a closure with exclusive access to one pooled instance for the given key.
    pub fn with<F, R>(&self, key: &K, f: F) -> Result<R, InferenceError>
    where
        F: FnOnce(&mut T) -> Result<R, InferenceError>,
    {
        let r = self.map.read();
        let pool = r.get(key).ok_or_else(|| InferenceError::ProcessingError {
            message: "Model not initialized for key".to_string(),
        })?;
        pool.with(f)
    }
}
