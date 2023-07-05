use nih_plug::prelude::*;

/// A simple low-pass filter implementation.
struct Filter {
    cutoff_freq: FloatParam,
    resonance: FloatParam,
    last_output: f32,
}

impl Filter {
    fn new() -> Self {
        Self {
            cutoff_freq: FloatParam::new(
                "Cutoff Frequency",
                1000.0,
                FloatRange::Linear {
                    min: 0.0,
                    max: 20000.0,
                },
            )
            .with_unit(" Hz"),
            resonance: FloatParam::new(
                "Resonance",
                0.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            ),
            last_output: 0.0,
        }
    }

    fn process(&mut self, buffer: &mut Buffer, sample_rate: f32) {
        let cutoff = self.cutoff_freq.value();
        let resonance = self.resonance.value();

        for channel in buffer.iter_mut() {
            for sample in channel {
                let input = *sample;
                let output = (1.0 - cutoff / sample_rate).clamp(0.0, 1.0) * self.last_output
                    + cutoff / sample_rate * input
                    - resonance * (input - self.last_output);
                *sample = output;
                self.last_output = output;
            }
        }
    }
}

impl Params for Filter {
    fn get_parameter(&self, id: u32) -> Option<Box<dyn Parameter>> {
        match id {
            0 => Some(Box::new(self.cutoff_freq.clone())),
            1 => Some(Box::new(self.resonance.clone())),
            _ => None,
        }
    }

    fn get_parameter_by_name(&self, name: &str) -> Option<Box<dyn Parameter>> {
        match name {
            "Cutoff Frequency" => Some(Box::new(self.cutoff_freq.clone())),
            "Resonance" => Some(Box::new(self.resonance.clone())),
            _ => None,
        }
    }
}

impl Plugin for Filter {
    const NAME: &'static str = "Filter";
    const VENDOR: &'static str = "Your Vendor";
    const UNIQUE_ID: &'static [u8; 4] = b"Filt";
    const VERSION: i32 = 1;
    const CATEGORY: PluginCategory = PluginCategory::Effect;

    const AUDIO_IO: AudioIOConfiguration = AudioIOConfiguration {
        inputs: 1,
        outputs: 1,
    };

    fn process(&mut self, buffer: &mut AudioBuffer<f32>) {
        let sample_rate = self.get_sample_rate();
        self.process(buffer, sample_rate);
    }
}

nih_export!(Filter);
