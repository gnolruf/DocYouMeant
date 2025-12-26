//! Singleton pattern macros for inference models.
//!
//! This module provides macros to reduce boilerplate when implementing the singleton
//! pattern for inference models. Models are wrapped in `Mutex` for thread-safe access
//! and stored in `OnceCell` for lazy initialization.
//!
//! # Available Macros
//!
//! - [`impl_simple_singleton!`] - For models with a single global instance
//! - [`impl_mode_singleton!`] - For models with separate instances per operating mode

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

/// Implements the singleton pattern for a mode-based inference model.
///
/// This macro generates:
/// - Static `OnceCell<Mutex<$model>>` instances for each mode variant
/// - A `get_or_init(mode)` method for eager initialization of a specific mode
/// - An `instance(mode)` method for accessing the singleton for a specific mode
///
/// # Generated Methods
///
/// - `pub fn get_or_init(mode: Mode) -> Result<(), InferenceError>` - Initializes singleton for mode
/// - `fn instance(mode: Mode) -> Result<&'static Mutex<Self>, InferenceError>` - Gets singleton for mode
#[macro_export]
macro_rules! impl_mode_singleton {
    (
        model: $model:ty,
        mode_type: $mode_type:ty,
        variants: {
            $($variant:ident => $instance:ident),+ $(,)?
        }
    ) => {
        $(
            static $instance: ::once_cell::sync::OnceCell<::std::sync::Mutex<$model>> =
                ::once_cell::sync::OnceCell::new();
        )+

        impl $model {
            /// Pre-initializes the singleton instance for the specified mode.
            ///
            /// Call this method during application startup to eagerly load models
            /// rather than waiting for the first inference request.
            ///
            /// # Arguments
            ///
            /// * `mode` - The operating mode to initialize
            ///
            /// # Errors
            ///
            /// Returns an [`InferenceError`] if model initialization fails.
            pub fn get_or_init(mode: $mode_type) -> Result<(), $crate::inference::InferenceError> {
                match mode {
                    $(
                        <$mode_type>::$variant => {
                            $instance.get_or_try_init(|| {
                                Self::new(<$mode_type>::$variant).map(::std::sync::Mutex::new)
                            })?;
                        }
                    )+
                }
                Ok(())
            }

            /// Returns a reference to the singleton instance for the specified mode.
            ///
            /// Initializes the instance if it hasn't been created yet.
            fn instance(mode: $mode_type) -> Result<&'static ::std::sync::Mutex<Self>, $crate::inference::InferenceError> {
                match mode {
                    $(
                        <$mode_type>::$variant => {
                            $instance.get_or_try_init(|| {
                                Self::new(<$mode_type>::$variant).map(::std::sync::Mutex::new)
                            })
                        }
                    )+
                }
            }
        }
    };
}

pub use impl_mode_singleton;
pub use impl_simple_singleton;
