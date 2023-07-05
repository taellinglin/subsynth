use atomic_float::AtomicF32;
use nih_plug::prelude::{util, Editor, GuiContext};
use nih_plug_iced::widgets as nih_widgets;
use nih_plug_iced::*;
use std::sync::Arc;
use std::time::Duration;

struct PolyModSynthInterface {
    gain: AtomicF32,
    amp_attack_ms: AtomicF32,
    amp_release_ms: AtomicF32,
    filter_cutoff: AtomicF32,
    filter_resonance: AtomicF32,
    filter_attack_ms: AtomicF32,
    filter_release_ms: AtomicF32,
}

impl Editor for PolyModSynthInterface {
    fn new() -> Self {
        Self {
            gain: AtomicF32::new(0.0),
            amp_attack_ms: AtomicF32::new(200.0),
            amp_release_ms: AtomicF32::new(100.0),
            filter_cutoff: AtomicF32::new(5000.0),
            filter_resonance: AtomicF32::new(0.5),
            filter_attack_ms: AtomicF32::new(100.0),
            filter_release_ms: AtomicF32::new(100.0),
        }
    }

    fn process_gui_event(&self, _event: GuiEvent) {}

    fn update_gui(&self, gui_context: &mut GuiContext) {
        gui_context.set_frame_rate(60);

        gui_context.show_window("Poly Mod Synth", |ui| {
            ui.vertical(|ui| {
                ui.label("Gain");
                nih_widgets::knob_f32(
                    ui,
                    &self.gain,
                    -36.0,
                    0.0,
                    |value| format!("{:.1} dB", value),
                    |value| util::db_to_gain(value),
                );

                ui.label("Amp Attack (ms)");
                nih_widgets::knob_f32(
                    ui,
                    &self.amp_attack_ms,
                    0.0,
                    2000.0,
                    |value| format!("{:.1} ms", value),
                    |value| value,
                );

                ui.label("Amp Release (ms)");
                nih_widgets::knob_f32(
                    ui,
                    &self.amp_release_ms,
                    0.0,
                    2000.0,
                    |value| format!("{:.1} ms", value),
                    |value| value,
                );

                ui.label("Filter Cutoff");
                nih_widgets::knob_f32(
                    ui,
                    &self.filter_cutoff,
                    20.0,
                    20000.0,
                    |value| format!("{:.0} Hz", value),
                    |value| value,
                );

                ui.label("Filter Resonance");
                nih_widgets::knob_f32(
                    ui,
                    &self.filter_resonance,
                    0.0,
                    1.0,
                    |value| format!("{:.2}", value),
                    |value| value,
                );

                ui.label("Filter Attack (ms)");
                nih_widgets::knob_f32(
                    ui,
                    &self.filter_attack_ms,
                    0.0,
                    2000.0,
                    |value| format!("{:.1} ms", value),
                    |value| value,
                );

                ui.label("Filter Release (ms)");
                nih_widgets::knob_f32(
                    ui,
                    &self.filter_release_ms,
                    0.0,
                    2000.0,
                    |value| format!("{:.1} ms", value),
                    |value| value,
                );
            });
        });
    }

    fn process_audio(&self, buffer: &mut [f32], sample_rate: f32) {
        // Process audio here using the parameter values
    }

    fn get_state(&self) -> EditorState {
        EditorState::new(Duration::from_secs(1)) // Use a default state duration of 1 second
    }
}

fn main() {
    let synth_interface = Arc::new(PolyModSynthInterface::new());
    run_editor(synth_interface);
}
