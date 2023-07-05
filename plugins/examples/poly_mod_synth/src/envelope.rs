use nih_plug::prelude::*;

/// An ADSR envelope implementation.
struct Envelope {
    attack_time: FloatParam,
    decay_time: FloatParam,
    sustain_level: FloatParam,
    release_time: FloatParam,
    state: EnvelopeState,
    level: f32,
    time: f32,
}

enum EnvelopeState {
    Idle,
    Attack,
    Decay,
    Sustain,
    Release,
}

impl Envelope {
    fn new() -> Self {
        Self {
            attack_time: FloatParam::new(
                "Attack Time",
                10.0,
                FloatRange::Linear {
                    min: 0.0,
                    max: 100.0,
                },
            )
            .with_unit(" ms"),
            decay_time: FloatParam::new(
                "Decay Time",
                10.0,
                FloatRange::Linear {
                    min: 0.0,
                    max: 100.0,
                },
            )
            .with_unit(" ms"),
            sustain_level: FloatParam::new(
                "Sustain Level",
                0.5,
                FloatRange::Linear {
                    min: 0.0,
                    max: 1.0,
                },
            ),
            release_time: FloatParam::new(
                "Release Time",
                10.0,
                FloatRange::Linear {
                    min: 0.0,
                    max: 100.0,
                },
            )
            .with_unit(" ms"),
            state: EnvelopeState::Idle,
            level: 0.0,
            time: 0.0,
        }
    }

    fn process(&mut self, buffer: &mut Buffer, sample_rate: f32) {
        let attack_time = self.attack_time.value() * 0.001; // Convert from ms to seconds
        let decay_time = self.decay_time.value() * 0.001; // Convert from ms to seconds
        let sustain_level = self.sustain_level.value();
        let release_time = self.release_time.value() * 0.001; // Convert from ms to seconds

        for sample in buffer.iter_mut() {
            match self.state {
                EnvelopeState::Idle => {
                    // Do nothing
                }
                EnvelopeState::Attack => {
                    self.level += 1.0 / (attack_time * sample_rate);
                    if self.level >= 1.0 {
                        self.state = EnvelopeState::Decay;
                        self.level = 1.0;
                    }
                }
                EnvelopeState::Decay => {
                    self.level -= (1.0 - sustain_level) / (decay_time * sample_rate);
                    if self.level <= sustain_level {
                        self.state = EnvelopeState::Sustain;
                        self.level = sustain_level;
                    }
                }
                EnvelopeState::Sustain => {
                    // Do nothing
                }
                EnvelopeState::Release => {
                    self.level -= sustain_level / (release_time * sample_rate);
                    if self.level <= 0.0 {
                        self.state = EnvelopeState::Idle;
                        self.level = 0.0;
                    }
                }
            }

            *sample *= self.level;
        }
    }
}

impl Params for Envelope {
    fn get_parameter(&self, id: u32) -> Option<Box<dyn Parameter>> {
        match id {
            0 => Some(Box::new(self.attack_time.clone())),
            1 => Some(Box::new(self.decay_time.clone())),
            2 => Some(Box::new(self.sustain_level.clone())),
            3 => Some(Box::new(self.release_time.clone())),
            _ => None,
        }
    }

    fn get_parameter_by_name(&self, name: &str) -> Option<Box<dyn Parameter>> {
        match name {
            "Attack Time" => Some(Box::new(self.attack_time.clone())),
            "Decay Time" => Some(Box::new(self.decay_time.clone())),
            "Sustain Level" => Some(Box::new(self.sustain_level.clone())),
            "Release Time" => Some(Box::new(self.release_time.clone())),
            _ => None,
        }
    }
}

impl Plugin for Envelope {
    const NAME: &'static str = "Envelope";
    const VENDOR: &'static str = "Your Vendor";
    const UNIQUE_ID: &'static [u8; 4] = b"Envl";
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

nih_export!(Envelope);
