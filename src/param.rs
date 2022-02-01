// nih-plug: plugins, but rewritten in Rust
// Copyright (C) 2022 Robbert van der Helm
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

//! TODO: Document how to use the [Param] trait. For the moment, just look at the gain example.

use std::collections::HashMap;
use std::fmt::Display;
use std::pin::Pin;
use std::sync::Arc;

pub type FloatParam = PlainParam<f32>;
pub type IntParam = PlainParam<i32>;

/// Re-export for use in the [Params] proc-macro.
pub use serde_json::from_str as deserialize_field;
/// Re-export for use in the [Params] proc-macro.
pub use serde_json::to_string as serialize_field;

/// A distribution for a parameter's range. Probably need to add some forms of skewed ranges and
/// maybe a callback based implementation at some point.
#[derive(Debug)]
pub enum Range<T> {
    Linear { min: T, max: T },
}

/// A normalizable range for type `T`, where `self` is expected to be a type `R<T>`. Higher kinded
/// types would have made this trait definition a lot clearer.
trait NormalizebleRange<T> {
    /// Normalize a plain, unnormalized value. Will be clamped to the bounds of the range if the
    /// normalized value exceeds `[0, 1]`.
    fn normalize(&self, plain: T) -> f32;

    /// Unnormalize a normalized value. Will be clamped to `[0, 1]` if the plain, unnormalized value
    /// would exceed that range.
    fn unnormalize(&self, normalized: f32) -> T;
}

/// A numerical parameter that's stored unnormalized. The range is used for the normalization
/// process.
pub struct PlainParam<T> {
    /// The field's current plain, unnormalized value. Should be initialized with the default value.
    /// Storing parameter values like this instead of in a single contiguous array is bad for cache
    /// locality, but it does allow for a much nicer declarative API.
    pub value: T,

    // // TODO: Add optional value smoothing using an Enum. This would need to include  at least
    // //       - `Smoothing::None`: Don't do any work, `value` is just the most recent vlaue in the
    // //         block
    // //       - `Smoothing::Smooth(f32)`: Automatically smooth to `f32` milliseconds. The host will
    // //         provide this as an iterator (would probably be much faster than precalculating
    // //         verything).
    // //       - `Smoothing::SampleAccurate(f32)`: Same as `Smooth`, but uses sample accurate
    // //         automation values if the host provides those instead of the last value.
    // //
    // //       And this would need to integrate nicely with the sample buffer iterator adapter when
    // //       that gets added
    // pub smoothed: Smoothing<T>,
    /// Optional callback for listening to value changes. The argument passed to this function is
    /// the parameter's new **plain** value. This should not do anything expensive as it may be
    /// called multiple times in rapid succession.
    ///
    /// To use this, you'll probably want to store an `Arc<Atomic*>` alongside the parmater in the
    /// parmaeters struct, move a clone of that `Arc` into this closure, and then modify that.
    pub value_changed: Option<Arc<dyn Fn(T) -> () + Send + Sync>>,

    /// The distribution of the parameter's values.
    pub range: Range<T>,
    /// The parameter's human readable display name.
    pub name: &'static str,
    /// The parameter value's unit, added after `value_to_string` if that is set.
    pub unit: &'static str,
    /// Optional custom conversion function from a plain **unnormalized** value to a string.
    pub value_to_string: Option<Arc<dyn Fn(T) -> String + Send + Sync>>,
    /// Optional custom conversion function from a string to a plain **unnormalized** value. If the
    /// string cannot be parsed, then this should return a `None`. If this happens while the
    /// parameter is being updated then the update will be canceled.
    pub string_to_value: Option<Arc<dyn Fn(&str) -> Option<T> + Send + Sync>>,
}

/// A simple boolean parmaeter.
pub struct BoolParam {
    /// The field's current, normalized value. Should be initialized with the default value.
    pub value: bool,

    /// Optional callback for listening to value changes. The argument passed to this function is
    /// the parameter's new value. This should not do anything expensive as it may be called
    /// multiple times in rapid succession.
    pub value_changed: Option<Arc<dyn Fn(bool) -> () + Send + Sync>>,

    /// The parameter's human readable display name.
    pub name: &'static str,
    /// Optional custom conversion function from a boolean value to a string.
    pub value_to_string: Option<Arc<dyn Fn(bool) -> String + Send + Sync>>,
    /// Optional custom conversion function from a string to a boolean value. If the string cannot
    /// be parsed, then this should return a `None`. If this happens while the parameter is being
    /// updated then the update will be canceled.
    pub string_to_value: Option<Arc<dyn Fn(&str) -> Option<bool> + Send + Sync>>,
}

/// Describes a single parmaetre of any type.
pub trait Param {
    /// The plain parameter type.
    type Plain;

    /// Set this parameter based on a string. Returns whether the updating succeeded. That can fail
    /// if the string cannot be parsed.
    ///
    /// TODO: After implementing VST3, check if we handle parsing failures correctly
    fn set_from_string(&mut self, string: &str) -> bool;

    /// Get the unnormalized value for this parameter.
    fn plain_value(&self) -> Self::Plain;

    /// Set this parameter based on a plain, unnormalized value.
    fn set_plain_value(&mut self, plain: Self::Plain);

    /// Get the normalized `[0, 1]` value for this parameter.
    fn normalized_value(&self) -> f32;

    /// Set this parameter based on a normalized value.
    fn set_normalized_value(&mut self, normalized: f32);

    /// Get the string representation for a normalized value. Used as part of the wrappers. Most
    /// plugin formats already have support for units, in which case it shouldn't be part of this
    /// string or some DAWs may show duplicate units.
    fn normalized_value_to_string(&self, normalized: f32, include_unit: bool) -> String;

    /// Get the string representation for a normalized value. Used as part of the wrappers.
    fn string_to_normalized_value(&self, string: &str) -> Option<f32>;

    /// Internal implementation detail for implementing [Params]. This should not be used directly.
    fn as_ptr(&self) -> ParamPtr;
}

impl<T> Default for PlainParam<T>
where
    T: Default,
    Range<T>: Default,
{
    fn default() -> Self {
        Self {
            value: T::default(),
            value_changed: None,
            range: Range::default(),
            name: "",
            unit: "",
            value_to_string: None,
            string_to_value: None,
        }
    }
}

impl Default for BoolParam {
    fn default() -> Self {
        Self {
            value: false,
            value_changed: None,
            name: "",
            value_to_string: None,
            string_to_value: None,
        }
    }
}

impl Default for Range<f32> {
    fn default() -> Self {
        Self::Linear { min: 0.0, max: 1.0 }
    }
}

impl Default for Range<i32> {
    fn default() -> Self {
        Self::Linear { min: 0, max: 1 }
    }
}

macro_rules! impl_plainparam {
    ($ty:ident, $plain:ty) => {
        impl Param for $ty {
            type Plain = $plain;

            fn set_from_string(&mut self, string: &str) -> bool {
                let value = match &self.string_to_value {
                    Some(f) => f(string),
                    // TODO: Check how Rust's parse function handles trailing garbage
                    None => string.parse().ok(),
                };

                match value {
                    Some(plain) => {
                        self.value = plain;
                        true
                    }
                    None => false,
                }
            }

            fn plain_value(&self) -> Self::Plain {
                self.value
            }

            fn set_plain_value(&mut self, plain: Self::Plain) {
                self.value = plain;
                if let Some(f) = &self.value_changed {
                    f(plain);
                }
            }

            fn normalized_value(&self) -> f32 {
                self.range.normalize(self.value)
            }

            fn set_normalized_value(&mut self, normalized: f32) {
                self.set_plain_value(self.range.unnormalize(normalized));
            }

            fn normalized_value_to_string(&self, normalized: f32, include_unit: bool) -> String {
                let value = self.range.unnormalize(normalized);
                match (&self.value_to_string, include_unit) {
                    (Some(f), true) => format!("{}{}", f(value), self.unit),
                    (Some(f), false) => format!("{}", f(value)),
                    (None, true) => format!("{}{}", value, self.unit),
                    (None, false) => format!("{}", value),
                }
            }

            fn string_to_normalized_value(&self, string: &str) -> Option<f32> {
                let value = match &self.string_to_value {
                    Some(f) => f(string),
                    // TODO: Check how Rust's parse function handles trailing garbage
                    None => string.parse().ok(),
                }?;

                Some(self.range.normalize(value))
            }

            fn as_ptr(&self) -> ParamPtr {
                ParamPtr::$ty(self as *const $ty as *mut $ty)
            }
        }
    };
}

impl_plainparam!(FloatParam, f32);
impl_plainparam!(IntParam, i32);

impl Param for BoolParam {
    type Plain = bool;

    fn set_from_string(&mut self, string: &str) -> bool {
        let value = match &self.string_to_value {
            Some(f) => f(string),
            None => Some(string.eq_ignore_ascii_case("true") || string.eq_ignore_ascii_case("on")),
        };

        match value {
            Some(plain) => {
                self.value = plain;
                true
            }
            None => false,
        }
    }

    fn plain_value(&self) -> Self::Plain {
        self.value
    }

    fn set_plain_value(&mut self, plain: Self::Plain) {
        self.value = plain;
        if let Some(f) = &self.value_changed {
            f(plain);
        }
    }

    fn normalized_value(&self) -> f32 {
        if self.value {
            1.0
        } else {
            0.0
        }
    }

    fn set_normalized_value(&mut self, normalized: f32) {
        self.set_plain_value(normalized > 0.5);
    }

    fn normalized_value_to_string(&self, normalized: f32, _include_unit: bool) -> String {
        let value = normalized > 0.5;
        match (value, &self.value_to_string) {
            (v, Some(f)) => format!("{}", f(v)),
            (true, None) => String::from("On"),
            (false, None) => String::from("Off"),
        }
    }

    fn string_to_normalized_value(&self, string: &str) -> Option<f32> {
        let value = match &self.string_to_value {
            Some(f) => f(string),
            None => Some(string.eq_ignore_ascii_case("true") || string.eq_ignore_ascii_case("on")),
        }?;

        Some(if value { 1.0 } else { 0.0 })
    }

    fn as_ptr(&self) -> ParamPtr {
        ParamPtr::BoolParam(self as *const BoolParam as *mut BoolParam)
    }
}

impl<T: Display + Copy> Display for PlainParam<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.value_to_string {
            Some(func) => write!(f, "{}{}", func(self.value), self.unit),
            None => write!(f, "{}{}", self.value, self.unit),
        }
    }
}

impl Display for BoolParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (self.value, &self.value_to_string) {
            (v, Some(func)) => write!(f, "{}", func(v)),
            (true, None) => write!(f, "On"),
            (false, None) => write!(f, "Off"),
        }
    }
}

impl NormalizebleRange<f32> for Range<f32> {
    fn normalize(&self, plain: f32) -> f32 {
        match &self {
            Range::Linear { min, max } => (plain - min) / (max - min),
        }
        .clamp(0.0, 1.0)
    }

    fn unnormalize(&self, normalized: f32) -> f32 {
        let normalized = normalized.clamp(0.0, 1.0);
        match &self {
            Range::Linear { min, max } => (normalized * (max - min)) + min,
        }
    }
}

impl NormalizebleRange<i32> for Range<i32> {
    fn normalize(&self, plain: i32) -> f32 {
        match &self {
            Range::Linear { min, max } => (plain - min) as f32 / (max - min) as f32,
        }
        .clamp(0.0, 1.0)
    }

    fn unnormalize(&self, normalized: f32) -> i32 {
        let normalized = normalized.clamp(0.0, 1.0);
        match &self {
            Range::Linear { min, max } => (normalized * (max - min) as f32).round() as i32 + min,
        }
    }
}

/// Describes a struct containing parameters and other persistent fields. The idea is that we can
/// have a normal struct containing [FloatParam] and other parameter types with attributes assigning
/// a unique identifier to each parameter. We can then build a mapping from those parameter IDs to
/// the parameters using the [Params::param_map] function. That way we can have easy to work with
/// JUCE-style parameter objects in the plugin without needing to manually register each parameter,
/// like you would in JUCE.
///
/// The other persistent parameters should be [PersistentField]s containing types that can be
/// serialized and deserialized with Serde.
///
/// # Safety
///
/// This implementation is safe when using from the wrapper because the plugin object needs to be
/// pinned, and it can never outlive the wrapper.
pub trait Params {
    /// Create a mapping from unique parameter IDs to parameters. This is done for every parameter
    /// field marked with `#[id = "stable_name"]`. Dereferencing the pointers stored in the values
    /// is only valid as long as this pinned object is valid.
    fn param_map(self: Pin<&Self>) -> HashMap<&'static str, ParamPtr>;

    /// All parameter IDs from `param_map`, in a stable order. This order will be used to display
    /// the parameters.
    fn param_ids(self: Pin<&Self>) -> &'static [&'static str];

    /// Serialize all fields marked with `#[persist = "stable_name"]` into a hash map containing
    /// JSON-representations of those fields so they can be written to the plugin's state and
    /// recalled later. This uses [serialize_field] under the hood.
    fn serialize_fields(&self) -> HashMap<String, String>;

    /// Restore all fields marked with `#[persist = "stable_name"]` from a hashmap created by
    /// [Self::serialize_fields]. All of thse fields should be wrapped in a [PersistentField] with
    /// thread safe interior mutability, like an `RwLock` or a `Mutex`. This gets called when the
    /// plugin's state is being restored. This uses [deserialize_field] under the hood.
    fn deserialize_fields(&self, serialized: &HashMap<String, String>);
}

/// Internal pointers to parameters. This is an implementation detail used by the wrappers.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ParamPtr {
    FloatParam(*mut FloatParam),
    IntParam(*mut IntParam),
    BoolParam(*mut BoolParam),
}

// These pointers only point to fields on pinned structs, and the caller always needs to make sure
// that dereferencing them is safe
unsafe impl Send for ParamPtr {}
unsafe impl Sync for ParamPtr {}

impl ParamPtr {
    /// Get the human readable name for this parameter.
    ///
    /// # Safety
    ///
    /// Calling this function is only safe as long as the object this `ParamPtr` was created for is
    /// still alive.
    pub unsafe fn name(&self) -> &'static str {
        match &self {
            ParamPtr::FloatParam(p) => (**p).name,
            ParamPtr::IntParam(p) => (**p).name,
            ParamPtr::BoolParam(p) => (**p).name,
        }
    }

    /// Get the unit label for this parameter.
    ///
    /// # Safety
    ///
    /// Calling this function is only safe as long as the object this `ParamPtr` was created for is
    /// still alive.
    pub unsafe fn unit(&self) -> &'static str {
        match &self {
            ParamPtr::FloatParam(p) => (**p).unit,
            ParamPtr::IntParam(p) => (**p).unit,
            ParamPtr::BoolParam(_) => "",
        }
    }

    /// Set this parameter based on a string. Returns whether the updating succeeded. That can fail
    /// if the string cannot be parsed.
    ///
    /// # Safety
    ///
    /// Calling this function is only safe as long as the object this `ParamPtr` was created for is
    /// still alive.
    pub unsafe fn set_from_string(&mut self, string: &str) -> bool {
        match &self {
            ParamPtr::FloatParam(p) => (**p).set_from_string(string),
            ParamPtr::IntParam(p) => (**p).set_from_string(string),
            ParamPtr::BoolParam(p) => (**p).set_from_string(string),
        }
    }

    /// Get the normalized `[0, 1]` value for this parameter.
    ///
    /// # Safety
    ///
    /// Calling this function is only safe as long as the object this `ParamPtr` was created for is
    /// still alive.
    pub unsafe fn normalized_value(&self) -> f32 {
        match &self {
            ParamPtr::FloatParam(p) => (**p).normalized_value(),
            ParamPtr::IntParam(p) => (**p).normalized_value(),
            ParamPtr::BoolParam(p) => (**p).normalized_value(),
        }
    }

    /// Set this parameter based on a normalized value.
    ///
    /// # Safety
    ///
    /// Calling this function is only safe as long as the object this `ParamPtr` was created for is
    /// still alive.
    pub unsafe fn set_normalized_value(&self, normalized: f32) {
        match &self {
            ParamPtr::FloatParam(p) => (**p).set_normalized_value(normalized),
            ParamPtr::IntParam(p) => (**p).set_normalized_value(normalized),
            ParamPtr::BoolParam(p) => (**p).set_normalized_value(normalized),
        }
    }

    /// Get the normalized value for a plain, unnormalized value, as a float. Used as part of the
    /// wrappers.
    ///
    /// # Safety
    ///
    /// Calling this function is only safe as long as the object this `ParamPtr` was created for is
    /// still alive.
    pub unsafe fn preview_normalized(&self, plain: f32) -> f32 {
        match &self {
            ParamPtr::FloatParam(p) => (**p).range.normalize(plain),
            ParamPtr::IntParam(p) => (**p).range.normalize(plain as i32),
            ParamPtr::BoolParam(_) => plain,
        }
    }

    /// Get the plain, unnormalized value for a normalized value, as a float. Used as part of the
    /// wrappers.
    ///
    /// # Safety
    ///
    /// Calling this function is only safe as long as the object this `ParamPtr` was created for is
    /// still alive.
    pub unsafe fn preview_plain(&self, normalized: f32) -> f32 {
        match &self {
            ParamPtr::FloatParam(p) => (**p).range.unnormalize(normalized),
            ParamPtr::IntParam(p) => (**p).range.unnormalize(normalized) as f32,
            ParamPtr::BoolParam(_) => normalized,
        }
    }

    /// Get the string representation for a normalized value. Used as part of the wrappers. Most
    /// plugin formats already have support for units, in which case it shouldn't be part of this
    /// string or some DAWs may show duplicate units.
    ///
    /// # Safety
    ///
    /// Calling this function is only safe as long as the object this `ParamPtr` was created for is
    /// still alive.
    pub unsafe fn normalized_value_to_string(&self, normalized: f32, include_unit: bool) -> String {
        match &self {
            ParamPtr::FloatParam(p) => (**p).normalized_value_to_string(normalized, include_unit),
            ParamPtr::IntParam(p) => (**p).normalized_value_to_string(normalized, include_unit),
            ParamPtr::BoolParam(p) => (**p).normalized_value_to_string(normalized, include_unit),
        }
    }

    /// Get the string representation for a normalized value. Used as part of the wrappers.
    ///
    /// # Safety
    ///
    /// Calling this function is only safe as long as the object this `ParamPtr` was created for is
    /// still alive.
    pub unsafe fn string_to_normalized_value(&self, string: &str) -> Option<f32> {
        match &self {
            ParamPtr::FloatParam(p) => (**p).string_to_normalized_value(string),
            ParamPtr::IntParam(p) => (**p).string_to_normalized_value(string),
            ParamPtr::BoolParam(p) => (**p).string_to_normalized_value(string),
        }
    }
}

/// The functinoality needed for persisting a field to the plugin's state, and for restoring values
/// when loading old state.
///
/// TODO: Modifying these fields (or any parameter for that matter) should mark the plugin's state
///       as dirty.
pub trait PersistentField<'a, T>: Send + Sync
where
    T: serde::Serialize + serde::Deserialize<'a>,
{
    fn set(&self, new_value: T);
    fn map<F, R>(&self, f: F) -> R
    where
        F: Fn(&T) -> R;
}

impl<'a, T> PersistentField<'a, T> for std::sync::RwLock<T>
where
    T: serde::Serialize + serde::Deserialize<'a> + Send + Sync,
{
    fn set(&self, new_value: T) {
        *self.write().expect("Poisoned RwLock on write") = new_value;
    }
    fn map<F, R>(&self, f: F) -> R
    where
        F: Fn(&T) -> R,
    {
        f(&self.read().expect("Poisoned RwLock on read"))
    }
}

impl<'a, T> PersistentField<'a, T> for parking_lot::RwLock<T>
where
    T: serde::Serialize + serde::Deserialize<'a> + Send + Sync,
{
    fn set(&self, new_value: T) {
        *self.write() = new_value;
    }
    fn map<F, R>(&self, f: F) -> R
    where
        F: Fn(&T) -> R,
    {
        f(&self.read())
    }
}

impl<'a, T> PersistentField<'a, T> for std::sync::Mutex<T>
where
    T: serde::Serialize + serde::Deserialize<'a> + Send + Sync,
{
    fn set(&self, new_value: T) {
        *self.lock().expect("Poisoned Mutex") = new_value;
    }
    fn map<F, R>(&self, f: F) -> R
    where
        F: Fn(&T) -> R,
    {
        f(&self.lock().expect("Poisoned Mutex"))
    }
}

macro_rules! impl_persistent_field_parking_lot_mutex {
    ($ty:ty) => {
        impl<'a, T> PersistentField<'a, T> for $ty
        where
            T: serde::Serialize + serde::Deserialize<'a> + Send + Sync,
        {
            fn set(&self, new_value: T) {
                *self.lock() = new_value;
            }
            fn map<F, R>(&self, f: F) -> R
            where
                F: Fn(&T) -> R,
            {
                f(&self.lock())
            }
        }
    };
}

impl_persistent_field_parking_lot_mutex!(parking_lot::Mutex<T>);
impl_persistent_field_parking_lot_mutex!(parking_lot::FairMutex<T>);

#[cfg(test)]
mod tests {
    use super::*;

    fn make_linear_float_range() -> Range<f32> {
        Range::Linear {
            min: 10.0,
            max: 20.0,
        }
    }

    fn make_linear_int_range() -> Range<i32> {
        Range::Linear { min: -10, max: 10 }
    }

    #[test]
    fn range_normalize_linear_float() {
        let range = make_linear_float_range();
        assert_eq!(range.normalize(17.5), 0.75);
    }

    #[test]
    fn range_normalize_linear_int() {
        let range = make_linear_int_range();
        assert_eq!(range.normalize(-5), 0.25);
    }

    #[test]
    fn range_unnormalize_linear_float() {
        let range = make_linear_float_range();
        assert_eq!(range.unnormalize(0.25), 12.5);
    }

    #[test]
    fn range_unnormalize_linear_int() {
        let range = make_linear_int_range();
        assert_eq!(range.unnormalize(0.75), 5);
    }

    #[test]
    fn range_unnormalize_linear_int_rounding() {
        let range = make_linear_int_range();
        assert_eq!(range.unnormalize(0.73), 5);
    }
}