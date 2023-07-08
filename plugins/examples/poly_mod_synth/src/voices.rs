use nih_plug::prelude::*;
use rand::Rng;
use rand_pcg::Pcg32;
use std::sync::Arc;

const NUM_VOICES: u32 = 16;
const MAX_BLOCK_SIZE: usize = 64;

const GAIN_POLY_MOD_ID: u32 = 0;

struct PolyModSynth {
    params: Arc<PolyModSynthParams>,
    prng: Pcg32,
    voices: [Option<Voice>; NUM_VOICES as usize],
    next_internal_voice_id: u64,
}

#[derive(Params)]
struct PolyModSynthParams {
    gain: FloatParam,
    amp_attack_ms: FloatParam,
    amp_release_ms: FloatParam,
}

struct Voice {
    voice_id: i32,
    channel: u8,
    note: u8,
    internal_voice_id: u64,
    velocity_sqrt: f32,
    phase: f32,
    phase_delta: f32,
    releasing: bool,
    amp_envelope: Smoother<f32>,
    voice_gain: Option<(f32, Smoother<f32>)>,
}

impl Default for PolyModSynth {
    fn default() -> Self {
        Self {
            params: Arc::new(PolyModSynthParams::default()),
            prng: Pcg32::new(420, 1337),
            voices: [None; NUM_VOICES as usize],
            next_internal_voice_id: 0,
        }
    }
}

impl Default for PolyModSynthParams {
    fn default() -> Self {
        Self {
            gain: FloatParam::new(
                "Gain",
                util::db_to_gain(-12.0),
                FloatRange::Linear {
                    min: util::db_to_gain(-36.0),
                    max: util::db_to_gain(0.0),
                },
            )
            .with_poly_modulation_id(GAIN_POLY_MOD_ID)
            .with_smoother(SmoothingStyle::Logarithmic(5.0))
            .with_unit(" dB"),
            amp_attack_ms: FloatParam::new(
                "Attack",
                200.0,
                FloatRange::Skewed {
                    min: 0.0,
                    max: 2000.0,
                    factor: FloatRange::skew_factor(-1.0),
                },
            )
            .with_step_size(0.1)
            .with_unit(" ms"),
            amp_release_ms: FloatParam::new(
                "Release",
                100.0,
                FloatRange::Skewed {
                    min: 0.0,
                    max: 2000.0,
                    factor: FloatRange::skew_factor(-1.0),
                },
            )
            .with_step_size(0.1)
            .with_unit(" ms"),
        }
    }
}

impl Plugin for PolyModSynth {
    const NAME: &'static str = "Poly Mod Synth";
    const VENDOR: &'static str = "Moist Plugins GmbH";
    const URL: &'static str = "https://youtu.be/dQw4w9WgXcQ";
    const EMAIL: &'static str = "<your-email-address>";

    type Instance = PolyModSynthInstance;

    fn new(_: f32, _: usize) -> Self {
        Self::default()
    }
}

struct PolyModSynthInstance {
    params: Arc<PolyModSynthParams>,
    gain: Smoother<f32>,
    voice_stealing: bool,
    voice_stealing_start_time: f64,
}

impl PluginInstance for PolyModSynthInstance {
    type Plugin = PolyModSynth;

    fn init(_: &SampleRate, _: usize) -> Self {
        Self {
            params: Arc::new(PolyModSynthParams::default()),
            gain: Smoother::new(0.0),
            voice_stealing: false,
            voice_stealing_start_time: 0.0,
        }
    }

    fn params(&self) -> Arc<dyn PluginParameters> {
        self.params.clone()
    }

    fn process<'a>(
        &mut self,
        events: impl Iterator<Item = Event<'a>>,
        outputs: &mut [impl OutputBuffer + 'a],
    ) {
        let block_size = outputs.iter().map(|out| out.samples()).min().unwrap_or(0);
        let num_channels = outputs.len();

        let mut voice_events = Vec::new();
        let mut mod_events = Vec::new();

        for event in events {
            match event.payload {
                EventPayload::NoteOn(event) => {
                    // Handle note-on event
                    // ...
                }
                EventPayload::NoteOff(event) => {
                    // Handle note-off event
                    // ...
                }
                EventPayload::PolyModulation(event) => {
                    // Handle polyphonic modulation event
                    // ...
                }
                EventPayload::MonoAutomation(event) => {
                    // Handle mono automation event
                    // ...
                }
                _ => (),
            }
        }

        // Process voices
        // ...
        // Process voices (continued)
        for output in outputs.iter_mut() {
            output.clear();
        }

        let mut voice_output_buffers = Vec::new();
        for _ in 0..NUM_VOICES {
            voice_output_buffers.push(vec![0.0; block_size]);
        }

        for voice in &mut self.voices {
            if let Some(voice) = voice {
                voice.process(
                    block_size,
                    &mut voice_output_buffers[voice.voice_id as usize],
                    &self.params,
                );
            }
        }

        // Mix voice output buffers to the final output
        for (output, voice_output) in outputs.iter_mut().zip(voice_output_buffers) {
            for (output_sample, voice_sample) in output.samples_mut().iter_mut().zip(voice_output) {
                *output_sample += voice_sample;
            }
        }

        // Clean up inactive voices
        for voice in &mut self.voices {
            if let Some(voice) = voice {
                if voice.is_inactive() {
                    voice.release();
                }
            }
        }
    }
}

// voices.rs

use nih_plug::prelude::*;

impl PolyModSynth {
    // Helper functions for the PolyModSynth struct
    // ...

    fn start_voice(&mut self, note: u8, velocity: u8) {
        // Logic for starting a new voice or retriggering an existing voice with note and velocity
        // information goes here
        let frequency = util::midi_note_to_freq(note);
        let voice = Voice::new(frequency, velocity);
        self.voices.push(voice);
    }

    fn release_voice(&mut self, note: u8) {
        // Logic for releasing the voice associated with the given note goes here
        let voice_idx = self.voices.iter().position(|voice| voice.note == note);
        if let Some(idx) = voice_idx {
            self.voices.remove(idx);
        }
    }

    fn generate_audio(&mut self, buffer: &mut Buffer<Stereo<f32>>) {
        // Logic for generating audio for the active voices and writing it to the buffer goes here
        for voice in &mut self.voices {
            for frame in buffer.frames_mut() {
                let sample = voice.generate_sample();
                frame.left += sample;
                frame.right += sample;
            }
        }
    }
}

impl Voice {
    // Helper functions for the Voice struct
    // ...
}

impl PluginParameters for PolyModSynthParams {
    // Implementation of PluginParameters trait
    // ...
}

// Additional helper functions and trait implementations
// ...
