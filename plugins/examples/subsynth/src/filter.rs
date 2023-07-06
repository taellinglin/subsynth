mod envelope;
use nih_plug::params::enums::{Enum, EnumDescription, EnumParam};
use envelope::ADSREnvelope;
use envelope::Envelope;
use std::f32::consts::PI;


#[derive(Enum, PartialEq)]
pub enum FilterType {
    Lowpass,
    Bandpass,
    Highpass,
    Notch,
    Statevariable,
}

impl EnumDescription for FilterType {
    fn ids() -> Option<&'static [&'static str]> {
        Some(&["lowpass", "bandpass", "highpass", "notch", "statevariable"])
    }

    fn to_index(self) -> usize {
        match self {
            FilterType::Lowpass => 0,
            FilterType::Bandpass => 1,
            FilterType::Highpass => 2,
            FilterType::Notch => 3,
            FilterType::Statevariable => 4,
        }
    }

    fn from_index(index: usize) -> Self {
        match index {
            0 => FilterType::Lowpass,
            1 => FilterType::Bandpass,
            2 => FilterType::Highpass,
            3 => FilterType::Notch,
            4 => FilterType::Statevariable,
            _ => panic!("Invalid index for FilterType"),
        }
    }
}

pub trait Filter: Send {
    fn process(&mut self, input: f32) -> f32;
    fn set_sample_rate(&mut self, sample_rate: f32);
}

pub struct FilterFactory;

impl FilterFactory {
    pub fn create_filter(
        filter_type: FilterType,
        cutoff: f32,
        cutoff_envelope: ADSREnvelope,
        resonance: f32,
        resonance_envelope: ADSREnvelope,
        sample_rate: f32,
    ) -> Box<dyn Filter> {
        match filter_type {
            FilterType::Lowpass => Box::new(LowpassFilter::new(
                cutoff,
                cutoff_envelope,
                resonance,
                resonance_envelope,
                sample_rate,
            )),
            FilterType::Bandpass => Box::new(BandpassFilter::new(
                cutoff,
                cutoff_envelope,
                resonance,
                resonance_envelope,
                sample_rate,
            )),
            FilterType::Highpass => Box::new(HighpassFilter::new(
                cutoff,
                cutoff_envelope,
                resonance,
                resonance_envelope,
                sample_rate,
            )),
            FilterType::Notch => Box::new(NotchFilter::new(
                cutoff,
                cutoff_envelope,
                resonance,
                resonance_envelope,
                sample_rate,
            )),
            FilterType::Statevariable => Box::new(StatevariableFilter::new(
                cutoff,
                cutoff_envelope,
                resonance,
                resonance_envelope,
                sample_rate,
            )),
        }
    }
}

pub struct HighpassFilter {
    cutoff: f32,
    resonance: f32,
    cutoff_envelope: ADSREnvelope,
    resonance_envelope: ADSREnvelope,
    sample_rate: f32,
    prev_input: f32,
    prev_output: f32,
}

impl HighpassFilter {
    pub fn new(
        cutoff: f32,
        cutoff_envelope: ADSREnvelope,
        resonance: f32,
        resonance_envelope: ADSREnvelope,
        sample_rate: f32,
    ) -> Self {
        HighpassFilter {
            cutoff,
            resonance,
            cutoff_envelope,
            resonance_envelope,
            sample_rate,
            prev_input: 0.0,
            prev_output: 0.0,
        }
    }

    pub fn process(&mut self, input: f32) -> f32 {
        let cutoff = self.cutoff * self.cutoff_envelope.get_value();
        let resonance = self.resonance * self.resonance_envelope.get_value();
        let c = 1.0 / (2.0 * PI * cutoff / self.sample_rate);
        let r = 1.0 - resonance;
        let output = c * (input - self.prev_input + r * self.prev_output);
        self.prev_input = input;
        self.prev_output = output;
        output
    }
}

pub struct BandpassFilter {
    cutoff: f32,
    resonance: f32,
    cutoff_envelope: ADSREnvelope,
    resonance_envelope: ADSREnvelope,
    sample_rate: f32,
    prev_input: f32,
    prev_output: f32,
}

impl BandpassFilter {
    pub fn new(
        cutoff: f32,
        cutoff_envelope: ADSREnvelope,
        resonance: f32,
        resonance_envelope: ADSREnvelope,
        sample_rate: f32,
    ) -> Self {
        BandpassFilter {
            cutoff,
            resonance,
            cutoff_envelope,
            resonance_envelope,
            sample_rate,
            prev_input: 0.0,
            prev_output: 0.0,
        }
    }

    pub fn process(&mut self, input: f32) -> f32 {
        let cutoff = self.cutoff * self.cutoff_envelope.get_value();
        let resonance = self.resonance * self.resonance_envelope.get_value();
        let c = 1.0 / (2.0 * PI * cutoff / self.sample_rate);
        let r = 1.0 - resonance;
        let output = c * (input - self.prev_output) + r * self.prev_output;
        self.prev_input = input;
        self.prev_output = output;
        output
    }
}

pub struct LowpassFilter {
    cutoff: f32,
    resonance: f32,
    cutoff_envelope: ADSREnvelope,
    resonance_envelope: ADSREnvelope,
    sample_rate: f32,
    prev_output: f32,
}

impl LowpassFilter {
    pub fn new(
        cutoff: f32,
        cutoff_envelope: ADSREnvelope,
        resonance: f32,
        resonance_envelope: ADSREnvelope,
        sample_rate: f32,
    ) -> Self {
        LowpassFilter {
            cutoff,
            resonance,
            cutoff_envelope,
            resonance_envelope,
            sample_rate,
            prev_output: 0.0,
        }
    }

    pub fn process(&mut self, input: f32) -> f32 {
        let cutoff = self.cutoff * self.cutoff_envelope.get_value();
        let resonance = self.resonance * self.resonance_envelope.get_value();
        let c = 1.0 / (2.0 * PI * cutoff / self.sample_rate);
        let r = resonance;
        let output = c * input + r * self.prev_output;
        self.prev_output = output;
        output
    }
}

pub struct NotchFilter {
    cutoff: f32,
    resonance: f32,
    cutoff_envelope: ADSREnvelope,
    resonance_envelope: ADSREnvelope,
    sample_rate: f32,
    prev_input: f32,
    prev_output: f32,
}

impl NotchFilter {
    pub fn new(
        cutoff: f32,
        cutoff_envelope: ADSREnvelope,
        resonance: f32,
        resonance_envelope: ADSREnvelope,
        sample_rate: f32,
    ) -> Self {
        NotchFilter {
            cutoff,
            resonance,
            cutoff_envelope,
            resonance_envelope,
            sample_rate,
            prev_input: 0.0,
            prev_output: 0.0,
        }
    }

    pub fn process(&mut self, input: f32) -> f32 {
        let cutoff = self.cutoff * self.cutoff_envelope.get_value();
        let resonance = self.resonance * self.resonance_envelope.get_value();
        let c = 1.0 / (2.0 * PI * cutoff / self.sample_rate);
        let r = resonance;
        let output = (input - self.prev_output) + r * (self.prev_input - self.prev_output);
        self.prev_input = input;
        self.prev_output = output;
        output
    }
}

pub struct StatevariableFilter {
    cutoff: f32,
    resonance: f32,
    cutoff_envelope: ADSREnvelope,
    resonance_envelope: ADSREnvelope,
    sample_rate: f32,
    prev_input: f32,
    lowpass_output: f32,
    highpass_output: f32,
    bandpass_output: f32,
}

impl StatevariableFilter {
    pub fn new(
        cutoff: f32,
        cutoff_envelope: ADSREnvelope,
        resonance: f32,
        resonance_envelope: ADSREnvelope,
        sample_rate: f32,
    ) -> Self {
        StatevariableFilter {
            cutoff,
            resonance,
            cutoff_envelope,
            resonance_envelope,
            sample_rate,
            prev_input: 0.0,
            lowpass_output: 0.0,
            highpass_output: 0.0,
            bandpass_output: 0.0,
        }
    }

    pub fn process(&mut self, input: f32) -> f32 {
        let cutoff = self.cutoff * self.cutoff_envelope.get_value();
        let resonance = self.resonance * self.resonance_envelope.get_value();

        let f = cutoff / self.sample_rate;
        let k = 2.0 * (1.0 - resonance);
        let q = 1.0 / (2.0 * resonance);

        let input_minus_hp = input - self.highpass_output;
        let lp_output = self.lowpass_output + f * self.bandpass_output;
        let hp_output = input_minus_hp - lp_output * q - self.bandpass_output;
        let bp_output = f * hp_output + self.bandpass_output;

        self.prev_input = input;
        self.lowpass_output = lp_output;
        self.highpass_output = hp_output;
        self.bandpass_output = bp_output;

        bp_output
    }
}
