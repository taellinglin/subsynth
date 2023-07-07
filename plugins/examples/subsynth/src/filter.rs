use nih_plug::params::enums::{Enum, EnumParam};
use enum_iterator::Sequence;

pub trait Envelope: Send {
    fn get_value(&mut self, dt: f32) -> f32;
    fn trigger(&mut self);
    fn release(&mut self);
}

pub struct ADSREnvelope {
    attack: f32,
    decay: f32,
    sustain: f32,
    release: f32,
    state: ADSREnvelopeState,
    time: f32,
}

enum ADSREnvelopeState {
    Idle,
    Attack,
    Decay,
    Sustain,
    Release,
}

impl ADSREnvelope {
    pub fn new(attack: f32, decay: f32, sustain: f32, release: f32) -> Self {
        ADSREnvelope {
            attack,
            decay,
            sustain,
            release,
            state: ADSREnvelopeState::Idle,
            time: 0.0,
        }
    }

    pub fn set_attack(&mut self, attack: f32) {
        self.attack = attack;
    }

    pub fn set_decay(&mut self, decay: f32) {
        self.decay = decay;
    }

    pub fn set_sustain(&mut self, sustain: f32) {
        self.sustain = sustain;
    }

    pub fn set_release(&mut self, release: f32) {
        self.release = release;
    }
}

impl Envelope for ADSREnvelope {
    fn get_value(&mut self, dt: f32) -> f32 {
        self.time += dt;
        match self.state {
            ADSREnvelopeState::Idle => 0.0,
            ADSREnvelopeState::Attack => {
                let attack_value = self.time / self.attack;
                if self.time >= self.attack {
                    self.state = ADSREnvelopeState::Decay;
                    self.time = 0.0;
                    1.0
                } else {
                    attack_value
                }
            }
            ADSREnvelopeState::Decay => {
                let decay_value = 1.0 - (1.0 - self.sustain) * (self.time / self.decay);
                if self.time >= self.decay {
                    self.state = ADSREnvelopeState::Sustain;
                    self.time = 0.0;
                    self.sustain
                } else {
                    decay_value
                }
            }
            ADSREnvelopeState::Sustain => self.sustain,
            ADSREnvelopeState::Release => {
                let release_value = self.sustain * (1.0 - (self.time / self.release));
                if self.time >= self.release {
                    self.state = ADSREnvelopeState::Idle;
                    self.time = 0.0;
                    0.0
                } else {
                    release_value
                }
            }
        }
    }

    fn trigger(&mut self) {
        self.state = ADSREnvelopeState::Attack;
        self.time = 0.0;
    }

    fn release(&mut self) {
        self.state = ADSREnvelopeState::Release;
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug, Enum, Sequence)]
pub enum FilterType {
    Lowpass,
    Bandpass,
    Highpass,
    Notch,
    Statevariable,
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
            FilterType::Lowpass => Box::new(LowpassFilter::new(cutoff, cutoff_envelope, resonance, resonance_envelope, sample_rate)),
            FilterType::Bandpass => Box::new(BandpassFilter::new(cutoff, cutoff_envelope, resonance, resonance_envelope, sample_rate)),
            FilterType::Highpass => Box::new(HighpassFilter::new(cutoff, cutoff_envelope, resonance, resonance_envelope, sample_rate)),
            FilterType::Notch => Box::new(NotchFilter::new(cutoff, cutoff_envelope, resonance, resonance_envelope, sample_rate)),
            FilterType::Statevariable => Box::new(StatevariableFilter::new(cutoff, cutoff_envelope, resonance, resonance_envelope, sample_rate)),
        }
    }
}

pub struct HighpassFilter {
    cutoff: f32,
    resonance: f32,
    cutoff_envelope: Box<dyn Envelope>,
    resonance_envelope: Box<dyn Envelope>,
    sample_rate: f32,
    prev_input: f32,
    prev_output: f32,
}

impl HighpassFilter {
    pub fn new(
        cutoff: f32,
        cutoff_envelope: Box<dyn Envelope>,
        resonance: f32,
        resonance_envelope: Box<dyn Envelope>,
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
}

impl Filter for HighpassFilter {
    fn process(&mut self, input: f32) -> f32 {
        let cutoff = self.cutoff * self.cutoff_envelope.get_value(self.cutoff_envelope.time);
        let resonance =
            self.resonance * self.resonance_envelope.get_value(self.resonance_envelope.time);

        let c = 1.0 / (2.0 * std::f32::consts::PI * cutoff / self.sample_rate);
        let r = 1.0 - resonance;
        let output = c * (input - self.prev_input + r * self.prev_output);
        self.prev_input = input;
        self.prev_output = output;
        output
    }

    fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
    }
}

pub struct BandpassFilter {
    cutoff: f32,
    resonance: f32,
    cutoff_envelope: Box<dyn Envelope>,
    resonance_envelope: Box<dyn Envelope>,
    sample_rate: f32,
    prev_input: f32,
    prev_output: f32,
}

impl BandpassFilter {
    pub fn new(
        cutoff: f32,
        cutoff_envelope: Box<dyn Envelope>,
        resonance: f32,
        resonance_envelope: Box<dyn Envelope>,
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
}

impl Filter for BandpassFilter {
    fn process(&mut self, input: f32) -> f32 {
        let cutoff = self.cutoff * self.cutoff_envelope.get_value(self.cutoff_envelope.time);
        let resonance =
            self.resonance * self.resonance_envelope.get_value(self.resonance_envelope.time);
        let c = 1.0 / (2.0 * std::f32::consts::PI * cutoff / self.sample_rate);
        let r = 1.0 - resonance;
        let output = c * (input - self.prev_output) + r * self.prev_output;
        self.prev_input = input;
        self.prev_output = output;
        output
    }

    fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
    }
}

pub struct LowpassFilter {
    cutoff: f32,
    resonance: f32,
    cutoff_envelope: Box<dyn Envelope>,
    resonance_envelope: Box<dyn Envelope>,
    sample_rate: f32,
    prev_output: f32,
}

impl LowpassFilter {
    pub fn new(
        cutoff: f32,
        cutoff_envelope: Box<dyn Envelope>,
        resonance: f32,
        resonance_envelope: Box<dyn Envelope>,
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
}

impl Filter for LowpassFilter {
    fn process(&mut self, input: f32) -> f32 {
        let cutoff = self.cutoff * self.cutoff_envelope.get_value(self.cutoff_envelope.time);
        let resonance =
            self.resonance * self.resonance_envelope.get_value(self.resonance_envelope.time);

        let c = 1.0 / (2.0 * std::f32::consts::PI * cutoff / self.sample_rate);
        let r = resonance;
        let output = c * input + r * self.prev_output;

        self.prev_output = output;
        output
    }

    fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
    }
}

pub struct NotchFilter {
    cutoff: f32,
    resonance: f32,
    cutoff_envelope: Box<dyn Envelope>,
    resonance_envelope: Box<dyn Envelope>,
    sample_rate: f32,
    prev_input: f32,
    prev_output: f32,
}

impl NotchFilter {
    pub fn new(
        cutoff: f32,
        cutoff_envelope: Box<dyn Envelope>,
        resonance: f32,
        resonance_envelope: Box<dyn Envelope>,
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
}

impl Filter for NotchFilter {
    fn process(&mut self, input: f32) -> f32 {
        let cutoff = self.cutoff * self.cutoff_envelope.get_value(self.cutoff_envelope.time);
        let resonance =
            self.resonance * self.resonance_envelope.get_value(self.resonance_envelope.time);
        let c = 1.0 / (2.0 * std::f32::consts::PI * cutoff / self.sample_rate);
        let r = resonance;
        let output = (input - self.prev_output) + r * (self.prev_input - self.prev_output);
        self.prev_input = input;
        self.prev_output = output;
        output
    }

    fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
    }
}

pub struct StatevariableFilter {
    cutoff: f32,
    resonance: f32,
    cutoff_envelope: Box<dyn Envelope>,
    resonance_envelope: Box<dyn Envelope>,
    sample_rate: f32,
    prev_input: f32,
    lowpass_output: f32,
    highpass_output: f32,
    bandpass_output: f32,
}

impl StatevariableFilter {
    pub fn new(
        cutoff: f32,
        cutoff_envelope: Box<dyn Envelope>,
        resonance: f32,
        resonance_envelope: Box<dyn Envelope>,
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
}

impl Filter for StatevariableFilter {
    fn process(&mut self, input: f32) -> f32 {
        let cutoff = self.cutoff * self.cutoff_envelope.get_value();
        let resonance =
            self.resonance * self.resonance_envelope.get_value();

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

    fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
    }
}



pub fn generate_filter(
    filter_type: FilterType,
    cutoff: f32,
    cutoff_envelope: f32,
    resonance: f32,
    resonance_envelope: f32,
    generated_sample: f32,
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