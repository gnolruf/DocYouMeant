//! Singleton pattern macros for inference models.
//!
//! This module provides macros to when implementing the singleton
//! pattern for inference models. Models are wrapped in `Mutex` for thread-safe access
//! and stored in `OnceCell` for lazy initialization.
//!
//! # Available Macros
//!
//! - [`impl_simple_singleton!`] - For models with a single global instance
//! - [`impl_keyed_singleton!`] - For models with separate instances per key (e.g., mode, language)
//! - [`impl_static_keyed_singleton!`] - For models with a fixed set of static variants
///
/// Implements the singleton pattern for a simple inference model.
///
/// This macro generates:
/// - A static `OnceCell<Mutex<$model>>` instance
/// - A `get_or_init()` method for eager initialization
/// - An `instance()` method for accessing the singleton
///
/// # Generated Methods
///
/// - `pub fn get_or_init() -> Result<(), InferenceError>` - Initializes the singleton
/// - `fn instance() -> Result<&'static Mutex<Self>, InferenceError>` - Gets the singleton reference
#[macro_export]
macro_rules! impl_simple_singleton {
    (
        model: $model:ty,
        instance: $instance:ident,
        init: $init:expr
    ) => {
        static $instance: ::once_cell::sync::OnceCell<::std::sync::Mutex<$model>> =
            ::once_cell::sync::OnceCell::new();

        impl $model {
            /// Pre-initializes the singleton instance.
            ///
            /// Call this method during application startup to eagerly load the model
            /// rather than waiting for the first inference request.
            ///
            /// # Errors
            ///
            /// Returns an [`InferenceError`] if model initialization fails.
            pub fn get_or_init() -> Result<(), $crate::inference::InferenceError> {
                $instance.get_or_try_init(|| {
                    let init_fn: fn() -> Result<Self, $crate::inference::InferenceError> = $init;
                    init_fn().map(::std::sync::Mutex::new)
                })?;
                Ok(())
            }

            /// Returns a reference to the singleton instance.
            ///
            /// Initializes the instance if it hasn't been created yet.
            fn instance(
            ) -> Result<&'static ::std::sync::Mutex<Self>, $crate::inference::InferenceError> {
                $instance.get_or_try_init(|| {
                    let init_fn: fn() -> Result<Self, $crate::inference::InferenceError> = $init;
                    init_fn().map(::std::sync::Mutex::new)
                })
            }
        }
    };
}

/// Implements the singleton pattern for a keyed inference model.
///
/// This macro generates:
/// - A static `OnceCell<RwLock<HashMap<K, Mutex<$model>>>>` for dynamic key-based singletons
/// - A `get_or_init(key)` method for eager initialization of a specific key
/// - An `instance(key)` method for accessing the singleton for a specific key
///
/// The key type must implement `Eq + Hash + Clone + Send + Sync + 'static`.
///
/// # Generated Methods
///
/// - `pub fn get_or_init(key: K) -> Result<(), InferenceError>` - Initializes singleton for key
/// - `fn with_instance<F, R>(key: K, f: F) -> Result<R, InferenceError>` - Executes closure with locked instance
#[macro_export]
macro_rules! impl_keyed_singleton {
    (
        model: $model:ty,
        key_type: $key_type:ty,
        instance: $instance:ident
    ) => {
        static $instance: ::once_cell::sync::OnceCell<
            ::std::sync::RwLock<::std::collections::HashMap<$key_type, ::std::sync::Mutex<$model>>>,
        > = ::once_cell::sync::OnceCell::new();

        impl $model {
            /// Pre-initializes the singleton instance for the specified key.
            ///
            /// Call this method during application startup to eagerly load models
            /// rather than waiting for the first inference request.
            ///
            /// # Arguments
            ///
            /// * `key` - The key (e.g., language, mode) to initialize
            ///
            /// # Errors
            ///
            /// Returns an [`InferenceError`] if model initialization fails.
            pub fn get_or_init(key: $key_type) -> Result<(), $crate::inference::InferenceError> {
                let map = $instance
                    .get_or_init(|| ::std::sync::RwLock::new(::std::collections::HashMap::new()));

                {
                    let read_guard = map.read().map_err(|e| {
                        $crate::inference::InferenceError::ProcessingError {
                            message: format!("Failed to acquire read lock: {e}"),
                        }
                    })?;
                    if read_guard.contains_key(&key) {
                        return Ok(());
                    }
                }

                let mut write_guard = map.write().map_err(|e| {
                    $crate::inference::InferenceError::ProcessingError {
                        message: format!("Failed to acquire write lock: {e}"),
                    }
                })?;

                if let ::std::collections::hash_map::Entry::Vacant(e) =
                    write_guard.entry(key.clone())
                {
                    let model = Self::new(key)?;
                    e.insert(::std::sync::Mutex::new(model));
                }

                Ok(())
            }

            /// Executes a closure with exclusive access to the model instance for the specified key.
            ///
            /// This method ensures thread-safe access to the model by acquiring the appropriate locks.
            /// The model is initialized if it doesn't exist yet.
            ///
            /// # Arguments
            ///
            /// * `key` - The key identifying which model instance to use
            /// * `f` - A closure that receives a mutable reference to the model
            ///
            /// # Returns
            ///
            /// The result of the closure, or an error if lock acquisition fails.
            ///
            /// # Errors
            ///
            /// Returns an [`InferenceError`] if:
            /// - Model initialization fails
            /// - Lock acquisition fails
            pub fn with_instance<F, R>(
                key: $key_type,
                f: F,
            ) -> Result<R, $crate::inference::InferenceError>
            where
                F: FnOnce(&mut Self) -> Result<R, $crate::inference::InferenceError>,
            {
                Self::get_or_init(key.clone())?;

                let map = $instance.get().ok_or_else(|| {
                    $crate::inference::InferenceError::ProcessingError {
                        message: "Instance map not initialized".to_string(),
                    }
                })?;

                let read_guard =
                    map.read()
                        .map_err(|e| $crate::inference::InferenceError::ProcessingError {
                            message: format!("Failed to acquire read lock: {e}"),
                        })?;

                let mutex = read_guard.get(&key).ok_or_else(|| {
                    $crate::inference::InferenceError::ProcessingError {
                        message: "Instance not found after initialization".to_string(),
                    }
                })?;

                let mut model = mutex.lock().map_err(|e| {
                    $crate::inference::InferenceError::ProcessingError {
                        message: format!("Failed to lock model instance: {e}"),
                    }
                })?;

                f(&mut model)
            }
        }
    };
}

/// Implements the singleton pattern for a mode-based inference model with static variants.
///
/// This macro generates:
/// - Static `OnceCell<Mutex<$model>>` instances for each mode variant
/// - A `get_or_init(mode)` method for eager initialization of a specific mode
/// - An `instance(mode)` method for accessing the singleton for a specific mode
///
/// Use this macro when you have a fixed, known set of variants at compile time.
/// For dynamic keys (like Language enum with many variants), use `impl_keyed_singleton!`.
///
/// # Generated Methods
///
/// - `pub fn get_or_init(key: K) -> Result<(), InferenceError>` - Initializes singleton for key
/// - `fn instance(key: K) -> Result<&'static Mutex<Self>, InferenceError>` - Gets singleton for key
#[macro_export]
macro_rules! impl_static_keyed_singleton {
    (
        model: $model:ty,
        key_type: $key_type:ty,
        variants: {
            $($variant:ident => $instance:ident),+ $(,)?
        }
    ) => {
        $(
            static $instance: ::once_cell::sync::OnceCell<::std::sync::Mutex<$model>> =
                ::once_cell::sync::OnceCell::new();
        )+

        impl $model {
            /// Pre-initializes the singleton instance for the specified key.
            ///
            /// Call this method during application startup to eagerly load models
            /// rather than waiting for the first inference request.
            ///
            /// # Arguments
            ///
            /// * `key` - The key (e.g., mode) to initialize
            ///
            /// # Errors
            ///
            /// Returns an [`InferenceError`] if model initialization fails.
            pub fn get_or_init(key: $key_type) -> Result<(), $crate::inference::InferenceError> {
                match key {
                    $(
                        <$key_type>::$variant => {
                            $instance.get_or_try_init(|| {
                                Self::new(<$key_type>::$variant).map(::std::sync::Mutex::new)
                            })?;
                        }
                    )+
                }
                Ok(())
            }

            /// Returns a reference to the singleton instance for the specified key.
            ///
            /// Initializes the instance if it hasn't been created yet.
            fn instance(key: $key_type) -> Result<&'static ::std::sync::Mutex<Self>, $crate::inference::InferenceError> {
                match key {
                    $(
                        <$key_type>::$variant => {
                            $instance.get_or_try_init(|| {
                                Self::new(<$key_type>::$variant).map(::std::sync::Mutex::new)
                            })
                        }
                    )+
                }
            }
        }
    };
}

pub use impl_keyed_singleton;
pub use impl_simple_singleton;
pub use impl_static_keyed_singleton;
