use nih_plug::prelude::*;
use std::sync::atomic::{AtomicI32, AtomicF32};

#[derive(Debug, Clone)]
struct Smoother<T> where T: Copy {
    value: AtomicT<T>,
    style: SmoothingStyle,
    target: AtomicT<T>,
    steps_left: AtomicI32,
    step_size: AtomicF32,
}

impl<T> Smoother<T> where T: Copy {
    fn none() -> Self {
        Self {
            value: AtomicT::new(Default::default()),
            style: SmoothingStyle::None,
            target: AtomicT::new(Default::default()),
            steps_left: AtomicI32::new(0),
            step_size: AtomicF32::new(0.0),
        }
    }

    fn new(style: SmoothingStyle) -> Self {
        Self {
            value: AtomicT::new(Default::default()),
            style,
            target: AtomicT::new(Default::default()),
            steps_left: AtomicI32::new(0),
            step_size: AtomicF32::new(0.0),
        }
    }

    fn reset(&self, target: T) {
        self.target.store(target);
        self.steps_left.store(0);
        self.step_size.store(0.0);
    }

    fn set_target(&self, sample_rate: f32, target: T) {
        if self.target.load() != target {
            let steps_left = (sample_rate / 1000.0 * self.style.time_to_exponential_constant()) as i32;
            let step_size = (target - self.target.load()) / steps_left as f32;
            self.steps_left.store(steps_left);
            self.step_size.store(step_size);
        }
    }

    fn next_block(&self, output: &mut [T], num_samples: usize) {
        match self.style {
            SmoothingStyle::None => {
                let value = self.target.load();
                for sample in output.iter_mut().take(num_samples) {
                    *sample = value;
                }
            }
            SmoothingStyle::Linear => {
                let steps_left = self.steps_left.load();
                let step_size = self.step_size.load();
                let mut value = self.value.load();
                for sample in output.iter_mut().take(num_samples) {
                    *sample = value;
                    if steps_left > 0 {
                        value += step_size;
                        self.steps_left.fetch_sub(1);
                    }
                }
                self.value.store(value);
            }
            SmoothingStyle::Exponential(time_constant_ms) => {
                let steps_left = self.steps_left.load();
                let step_size = self.step_size.load();
                let mut value = self.value.load();
                for sample in output.iter_mut().take(num_samples) {
                    *sample = value;
                    if steps_left > 0 {
                        value += step_size * value;
                        self.steps_left.fetch_sub(1);
                    }
                }
                self.value.store(value);
            }
            SmoothingStyle::Logarithmic(log_base) => {
                let steps_left = self.steps_left.load();
                let step_size = self.step_size.load();
                let mut value = self.value.load();
                for sample in output.iter_mut().take(num_samples) {
                    *sample = value;
                    if steps_left > 0 {
                        value += step_size * (value.log(log_base) + 1.0);
                        self.steps_left.fetch_sub(1);
                    }
                }
                self.value.store(value);
            }
        }
    }

    fn preview_modulated(&self, normalized_offset: f32) -> T {
        match self.style {
            SmoothingStyle::None => self.target.load(),
            SmoothingStyle::Linear => {
                let steps_left = self.steps_left.load();
                let step_size = self.step_size.load();
                let mut value = self.value.load();
                for _ in 0..steps_left {
                    value += step_size;
                }
                value + normalized_offset * step_size
            }
            SmoothingStyle::Exponential(time_constant_ms) => {
                let steps_left = self.steps_left.load();
                let step_size = self.step_size.load();
                let mut value = self.value.load();
                for _ in 0..steps_left {
                    value += step_size * value;
                }
                value * (1.0 + normalized_offset * step_size)
            }
            SmoothingStyle::Logarithmic(log_base) => {
                let steps_left = self.steps_left.load();
                let step_size = self.step_size.load();
                let mut value = self.value.load();
                for _ in 0..steps_left {
                    value += step_size * (value.log(log_base) + 1.0);
                }
                value * (1.0 + normalized_offset * step_size)
            }
        }
    }
}

#[derive(Debug, Clone)]
enum SmoothingStyle {
    None,
    Linear,
    Exponential(f32),
    Logarithmic(f32),
}

#[derive(Default)]
struct FloatParam {
    // ...
    smoothed: Smoother<f32>,
    // ...
}
