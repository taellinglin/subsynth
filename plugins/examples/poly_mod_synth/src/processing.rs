use nih_plug::prelude::*;
use rand_pcg::Pcg32;
use std::sync::Arc;

use super::{PolyModSynth, Voice};

impl PolyModSynth {
    // Method to start a new voice or retrigger an existing voice with the note and velocity information
    fn start_voice(&mut self, note: u8, velocity: u8) {
        // Logic for starting a new voice or retriggering an existing voice with note and velocity
        // information goes here
    }

    // Method to release the voice associated with the given note
    fn release_voice(&mut self, note: u8) {
        // Logic for releasing the voice associated with the given note goes here
    }

    // Method to generate audio for the active voices and write it to the buffer
    fn generate_audio(&mut self, buffer: &mut Buffer<Stereo<f32>>) {
        // Logic for generating audio for the active voices and writing it to the buffer goes here
    }
}
