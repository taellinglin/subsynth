mod waveform;
mod editor;
mod filter;

use std::ops::Add;
use std::ops::Mul;
use std::ops::Sub;
use nih_plug::prelude::*;
use rand::Rng;
use rand_pcg::Pcg32;
use std::sync::Arc;
use waveform::Waveform;
use waveform::generate_waveform;
use filter::{NotchFilter, BandpassFilter, HighpassFilter, LowpassFilter, StatevariableFilter};
use filter::{Filter, FilterType, FilterFactory, Envelope, ADSREnvelope};


use filter::generate_filter;

use nih_plug_iced::IcedState;
use nih_plug::params::enums::EnumParam;

const NUM_VOICES: u32 = 16;
const MAX_BLOCK_SIZE: usize = 64;
const GAIN_POLY_MOD_ID: u32 = 0;

#[derive(Debug, Clone)]
struct Smoother<T> {
    value: T,
    target: T,
    time_constant: f32,
}

impl<T> Smoother<T>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<f32, Output = T>,
{
    pub fn new(value: T, time_constant: f32) -> Self {
        Smoother {
            value,
            target: value,
            time_constant,
        }
    }

    pub fn set_target(&mut self, target: T) {
        self.target = target;
    }

    pub fn next_value(&mut self, time: f32) -> T {
        let alpha = (-1.0 / (time * self.time_constant)).exp();
        self.value = self.value * alpha + self.target * (1.0 - alpha);
        self.value
    }
}
struct SubSynth {
    params: Arc<SubSynthParams>,
    prng: Pcg32,
    voices: [Option<Voice>; NUM_VOICES as usize],
    next_internal_voice_id: u64,
    
}

#[derive(Params)]
struct SubSynthParams {
    #[persist = "editor-state"]
    editor_state: Arc<IcedState>,
    #[id = "gain"]
    gain: FloatParam,
    #[id = "amp_atk"]
    amp_attack_ms: FloatParam,
    #[id = "amp_rel"]
    amp_release_ms: FloatParam,
    #[id = "waveform"]
    waveform: EnumParam<Waveform>,
    // New parameters for ADSR envelope
    #[id = "amp_dec"]
    amp_decay_ms: FloatParam,
    #[id = "amp_sus"]
    amp_sustain: FloatParam,
    #[id = "filter_cut_atk"]
    filter_cut_attack_ms: FloatParam,
    #[id = "filter_cut_dec"]
    filter_cut_decay_ms: FloatParam,
    #[id = "filter_cut_sus"]
    filter_cut_sustain: FloatParam,
    #[id = "filter_cut_rel"]
    filter_cut_release_ms: FloatParam,
    #[id = "filter_res_atk"]
    filter_res_attack_ms: FloatParam,
    #[id = "filter_res_dec"]
    filter_res_decay_ms: FloatParam,
    #[id = "filter_res_sus"]
    filter_res_sustain: FloatParam,
    #[id = "filter_res_rel"]
    filter_res_release_ms: FloatParam,
    #[id = "filter_type"]
    filter_type: EnumParam<FilterType>,
    #[id = "filter_cut"]
    filter_cut: FloatParam,
    #[id = "filter_res"]
    filter_res: FloatParam,
}

#[derive(Debug, Clone)]
pub struct Voice {
    voice_id: i32,
    internal_voice_id: u32,
    channel: u8,
    note: u8,
    velocity_sqrt: f32,
    voice_gain: Option<(f32, Smoother<f32>)>,
    phase: f32,
    phase_delta: f32,
    releasing: bool,
    amp_envelope: ADSREnvelope,
    channel_gain_left: f32, // Added field
    channel_gain_right: f32, // Added field
    cutoff_envelope: ADSREnvelope,
    resonance_envelope: ADSREnvelope,
    cutoff: Option<(f32, Smoother<f32>)>,
    resonance: Option<(f32, Smoother<f32>)>,
    pitch_bend: f32,
    filter: Option<Filter>,
}



impl Default for SubSynth {
    fn default() -> Self {
        Self {
            
            params: Arc::new(SubSynthParams::default()),

            prng: Pcg32::new(420, 1337),
            voices: [0; NUM_VOICES as usize].map(|_| None),
            next_internal_voice_id: 0,
        }
    }
}

impl Default for SubSynthParams {
    fn default() -> Self {
        Self {
            editor_state: editor::default_state(),
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
            .with_unit(" dB")
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .with_string_to_value(formatters::s2v_f32_gain_to_db()),
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
            waveform: EnumParam::new("Waveform", Waveform::Sine),
            amp_decay_ms: FloatParam::new(
                "Decay",
                200.0,
                FloatRange::Skewed {
                    min: 0.0,
                    max: 2000.0,
                    factor: FloatRange::skew_factor(-1.0),
                },
            )
            .with_step_size(0.1)
            .with_unit(" ms"),
            amp_sustain: FloatParam::new(
                "Sustain",
                1000.0,
                FloatRange::Skewed {
                    min: 0.0,
                    max: 5000.0,
                    factor: FloatRange::skew_factor(-1.0),
                },
            )
            .with_step_size(0.1)
            .with_unit(" ms"),
            filter_type: EnumParam::new("Filter Type", FilterType::Lowpass),
            filter_cut: FloatParam::new(
                "Filter Cut",
                10000.0,
                FloatRange::Linear {
                    min: 20.0,
                    max: 20000.0,
                },
            )
            .with_unit(" Hz"),
            filter_res: FloatParam::new(
                "Filter Resonance",
                0.5,
                FloatRange::Linear {
                    min: 0.0,
                    max: 1.0,
                },
            )
            .with_unit(" Q"),
            filter_cut_attack_ms: FloatParam::new(
                "Filter Cut Attack",
                200.0,
                FloatRange::Skewed {
                    min: 0.0,
                    max: 2000.0,
                    factor: FloatRange::skew_factor(-1.0),
                },
            )
            .with_step_size(0.1)
            .with_unit(" ms"),
            filter_cut_decay_ms: FloatParam::new(
                "Filter Cut Decay",
                200.0,
                FloatRange::Skewed {
                    min: 0.0,
                    max: 2000.0,
                    factor: FloatRange::skew_factor(-1.0),
                },
            )
            .with_step_size(0.1)
            .with_unit(" ms"),
            filter_cut_sustain: FloatParam::new(
                "Filter Cut Sustain",
                1000.0,
                FloatRange::Skewed {
                    min: -1.0,
                    max: 1.0,
                    factor: FloatRange::skew_factor(-1.0),
                },
            )
            .with_step_size(0.01)
            .with_unit(" +/-"),
            filter_cut_release_ms: FloatParam::new(
                "Filter Cut Release",
                100.0,
                FloatRange::Skewed {
                    min: 0.0,
                    max: 2000.0,
                    factor: FloatRange::skew_factor(-1.0),
                },
            )
            .with_step_size(0.1)
            .with_unit(" ms"),
            filter_res_attack_ms: FloatParam::new(
                "Filter Resonance Attack",
                200.0,
                FloatRange::Skewed {
                    min: 0.0,
                    max: 2000.0,
                    factor: FloatRange::skew_factor(-1.0),
                },
            )
            .with_step_size(0.1)
            .with_unit(" ms"),
            filter_res_decay_ms: FloatParam::new(
                "Filter Resonance Decay",
                200.0,
                FloatRange::Skewed {
                    min: 0.0,
                    max: 2000.0,
                    factor: FloatRange::skew_factor(-1.0),
                },
            )
            .with_step_size(0.1)
            .with_unit(" ms"),
            filter_res_sustain: FloatParam::new(
                "Filter Resonance Sustain",
                1000.0,
                FloatRange::Skewed {
                    min: -1.0,
                    max: 1.0,
                    factor: FloatRange::skew_factor(-1.0),
                },
            )
            .with_step_size(0.01)
            .with_unit("+/-1"),
            filter_res_release_ms: FloatParam::new(
                "Filter Resonance Decay",
                200.0,
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



impl Plugin for SubSynth {
    const NAME: &'static str = "SubSynthBeta";
    const VENDOR: &'static str = "LingYue Synth";
    const URL: &'static str = "https://taellinglin.art";
    const EMAIL: &'static str = "taellinglin@gmail.com";

    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(2),
        main_output_channels: NonZeroU32::new(2),
        ..AudioIOLayout::const_default()
    }];

    const MIDI_INPUT: MidiConfig = MidiConfig::Basic;
    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }
    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        editor::create(
            self.params.clone(),
            self.params.editor_state.clone(),
        )
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        // After `PEAK_METER_DECAY_MS` milliseconds of pure silence, the peak meter's value should
        // have dropped by 12 dB

        true
    }

    fn reset(&mut self) {
        self.prng = Pcg32::new(420, 1337);

        self.voices.fill(None);
        self.next_internal_voice_id = 0;
    }

}
    
   
    


impl SubSynth {
    fn get_voice_idx(&mut self, voice_id: i32) -> Option<usize> {
        self.voices
            .iter_mut()
            .position(|voice| matches!(voice, Some(voice) if voice.voice_id == voice_id))
    }

    fn start_voice(
        &mut self,
        context: &mut impl ProcessContext<Self>,
        sample_offset: u32,
        voice_id: Option<i32>,
        channel: u8,
        note: u8,
    ) -> &mut Voice {
        let params = self.params.as_ref();
        let new_voice = Voice {
            voice_id: voice_id.unwrap_or_else(|| compute_fallback_voice_id(note, channel)),
            internal_voice_id: self.next_internal_voice_id,
            channel,
            note,
            velocity_sqrt: 1.0,
            voice_gain,
            phase: 0.0,
            phase_delta: 0.0,
            releasing: false,
            amp_envelope: ADSREnvelope::new(0.0,1.0, 0.25, 0.25),
            channel_gain_left: voice_gain * (1.0 - params.channel_panning.value()) * (1.0 - params.panning.value()),
            channel_gain_right: voice_gain * params.channel_panning.value() * params.panning.value(),

            cutoff_envelope: ADSREnvelope::new(
                params.filter_cut_attack_ms.value(),
                params.filter_cut_decay_ms.value(),
                params.filter_cut_sustain.value(),
                params.filter_cut_release_ms.value(),
            ),
            resonance_envelope: ADSREnvelope::new(
                params.filter_res_attack_ms.value(),
                params.filter_res_decay_ms.value(),
                params.filter_res_sustain.value(),
                params.filter_res_release_ms.value(),
            ),

            cutoff: Some((params.filter_cut.value(), Smoother::new(0.0, 0.0))),,
            resonance: Some((params.filter_res.value(), Smoother::new(0.0, 0.0))),
            pitch_bend: f32,

            filter: None,
        };
        self.next_internal_voice_id = self.next_internal_voice_id.wrapping_add(1);

        match self.voices.iter().position(|voice| voice.is_none()) {
            Some(free_voice_idx) => {
                self.voices[free_voice_idx] = Some(new_voice);
                return self.voices[free_voice_idx].as_mut().unwrap();
            }
            None => {
                let oldest_voice = unsafe {
                    self.voices
                        .iter_mut()
                        .min_by_key(|voice| voice.as_ref().unwrap_unchecked().internal_voice_id)
                        .unwrap_unchecked()
                };
                {
                    let oldest_voice = oldest_voice.as_ref().unwrap();
                    context.send_event(NoteEvent::NoteOff {
                        timing: sample_offset,
                        voice_id: Some(oldest_voice.voice_id),
                        channel: oldest_voice.channel,
                        note: oldest_voice.note,
                        velocity: 0.0,
                    });
                }

                *oldest_voice = Some(new_voice);
                return oldest_voice.as_mut().unwrap();
            }
        }
    }
    pub fn process_events(
        &mut self,
        context: &mut impl ProcessContext<Self>,
        block_start: usize,
        block_end: usize,
    ) {
        let sample_rate = context.transport().sample_rate;
        let mut next_event = context.next_event();
    
        // Dereference self.params once
        let params = self.params.as_ref();
    
        'events: loop {
            match next_event {
                Some(event) if (event.timing() as usize) <= block_start => {
                    match event {
                        NoteEvent::NoteOn {
                            timing,
                            voice_id,
                            channel,
                            note,
                            velocity,
                        } => {
                            let initial_phase: f32 = self.prng.gen();
                            // Calculate amplitude for each channel based on velocity and panning
                            let amp_left = velocity as f32 * (1.0 - voice.channel_panning);
                            let amp_right = velocity as f32 * voice.channel_panning;

                            let mut amp_envelope = ADSREnvelope::new(
                                params.amp_attack_ms.value(),
                                params.amp_decay_ms.value(),
                                params.amp_sustain.value(),
                                params.amp_release_ms.value(),
                            );
                            amp_envelope.trigger();

                            amp_envelope.next_block(&mut voice_amp_envelope, block_end - block_start);
                            let amp_envelope_left = voice_amp_envelope;

                            amp_envelope.next_block(&mut voice_amp_envelope, block_end - block_start);
                            let amp_envelope_right = voice_amp_envelope;

                            // Start the voice and set parameters
                            let voice = self.start_voice(
                                context,
                                timing,
                                voice_id,
                                channel,
                                note,
                            );
                            voice.velocity_sqrt = velocity.sqrt();
                            voice.phase = initial_phase;
                            voice.phase_delta = util::midi_note_to_freq(note) / sample_rate;

                            // Set channel gains
                            voice.channel_gain_left = match &voice.voice_gain {
                                Some((value, _)) => value * amp_left * amp_envelope_left,
                                None => 0.0, // Handle the case when voice.voice_gain is None
                            };
                            
                            voice.channel_gain_right = match &voice.voice_gain {
                                Some((value, _)) => value * amp_right * amp_envelope_right,
                                None => 0.0, // Handle the case when voice.voice_gain is None
                            };
                            

                            voice.cutoff_envelope.trigger(); // Trigger cutoff envelope
                            voice.resonance_envelope.trigger(); // Trigger resonance envelope

                            // Set envelope and filter parameters
                            // ...

    
                            // Set envelope and filter parameters
                            voice.amp_envelope.set_target(sample_rate, amp_left); // Set target amplitude based on envelope
                            voice.cutoff_envelope.set_attack(params.filter_cut_attack_ms.value()); // Set attack time for cutoff envelope
                            voice.cutoff_envelope.set_decay(params.filter_cut_decay_ms.value()); // Set decay time for cutoff envelope
                            voice.cutoff_envelope.set_sustain(params.filter_cut_sustain.value()); // Set sustain level for cutoff envelope
                            voice.cutoff_envelope.set_release(params.filter_cut_release_ms.value()); // Set release time for cutoff envelope
                            voice.resonance_envelope.set_attack(params.filter_res_attack_ms.value()); // Set attack time for resonance envelope
                            voice.resonance_envelope.set_decay(params.filter_res_decay_ms.value()); // Set decay time for resonance envelope
                            voice.resonance_envelope.set_sustain(params.filter_res_sustain.value()); // Set sustain level for resonance envelope
                            voice.resonance_envelope.set_release(params.filter_res_release_ms.value()); // Set release time for resonance envelope
                        }
                        // ... other event cases ...
                    };
    
                    next_event = context.next_event();
                }
                Some(event) if (event.timing() as usize) < block_end => {
                    break 'events;
                }
                _ => break 'events,
            }
        }
    }
    fn process_voices(
        &mut self,
        block_start: usize,
        block_end: usize,
        output: &mut [f32; MAX_BLOCK_SIZE],
        context: &impl ProcessContext<Self>,
    ) {
        let sample_rate = context.transport().sample_rate;
        let panning = self.params.panning.value();
    
        let mut gain = [0.0; MAX_BLOCK_SIZE];
        let mut voice_gain = [0.0; MAX_BLOCK_SIZE];
        let mut voice_amp_envelope = [0.0; MAX_BLOCK_SIZE];
        self.params.gain.smoothed.next_block(&mut gain, block_end - block_start);
    
        for voice in self.voices.iter_mut().filter_map(|v| v.as_mut()) {
            let gain = match &voice.voice_gain {
                Some((_, smoother)) => {
                    smoother.next_block(&mut voice_gain, block_end - block_start);
                    &voice_gain
                }
                None => &gain,
            };
    
            voice
                .amp_envelope
                .next_block(&mut voice_amp_envelope, block_end - block_start);
    
            for (value_idx, sample_idx) in (block_start..block_end).enumerate() {
                let amp = voice.velocity_sqrt * gain[value_idx] * voice_amp_envelope[value_idx];
    
                // Generate waveform
                let waveform = self.params.waveform.value();
                let generated_sample = generate_waveform(waveform, voice.phase) * amp;
    
                // Apply filter
                let filter_type = self.params.filter_type.value();
                let cutoff = self.params.filter_cut.value();
                let resonance = self.params.filter_res.value();
                let cutoff_envelope = voice.cutoff_envelope.get_next(sample_idx as f32);
                let resonance_envelope = voice.resonance_envelope.get_next(sample_idx as f32);
                let sample_rate = context.transport().sample_rate;
    
                let mut filtered_sample = generate_filter(
                    filter_type,
                    cutoff,
                    cutoff_envelope,
                    resonance,
                    resonance_envelope,
                    generated_sample,
                    sample_rate,
                );
    
                // Apply any other processing or effects to the filtered sample
                // ...
    
                // Add the processed sample to the output buffer
                output[0][sample_idx] += filtered_sample * voice.channel_gain_left;
                output[1][sample_idx] += filtered_sample * voice.channel_gain_right;
    
                // Update the phase for the next sample
                voice.phase += voice.phase_delta;
                voice.phase -= voice.phase.floor();
    
                // Check if the voice has finished and release it if necessary
                if voice.amp_envelope.is_released() {
                    self.release_voice(context, voice);
                }
            }
        }
    }
    
    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let num_samples = buffer.samples();
        let mut output = buffer.as_mut_slice();
    
        let mut block_start: usize = 0;
        let mut block_end: usize = MAX_BLOCK_SIZE.min(num_samples);
    
        while block_start < num_samples {
            self.process_events(context, block_start, block_end);
    
            output[0][block_start..block_end].fill(0.0);
            output[1][block_start..block_end].fill(0.0);
    
            for voice in self.voices.iter_mut().filter_map(|v| v.as_mut()) {
                self.process_voices(context, block_start, block_end, voice, &mut output);
            }
    
            // Move to the next block
            block_start = block_end;
            block_end = (block_start + MAX_BLOCK_SIZE).min(num_samples);
        }
    
        ProcessStatus::Normal
    }
    
    fn start_release_for_voices(
        &mut self,
        _sample_rate: f32,
        voice_id: Option<i32>,
        channel: u8,
        note: u8,
    ) {
        for voice in self.voices.iter_mut() {
            match voice {
                Some(Voice {
                    voice_id: candidate_voice_id,
                    channel: candidate_channel,
                    note: candidate_note,
                    releasing,
                    amp_envelope,
                    cutoff_envelope,
                    resonance_envelope,
                    ..
                }) if voice_id == Some(*candidate_voice_id)
                    || (channel == *candidate_channel && note == *candidate_note) =>
                {
                    *releasing = true;
                    amp_envelope.release();
    
                    // Trigger cutoff and resonance release envelopes
                    cutoff_envelope.release();
                    resonance_envelope.release();
    
                    if voice_id.is_some() {
                        return;
                    }
                }
                _ => (),
            }
        }
    }
    
    
    fn choke_voices(
        &mut self,
        context: &mut impl ProcessContext<Self>,
        sample_offset: u32,
        voice_id: Option<i32>,
        channel: u8,
        note: u8,
    ) {
        for voice in self.voices.iter_mut() {
            match voice {
                Some(Voice {
                    voice_id: candidate_voice_id,
                    channel: candidate_channel,
                    note: candidate_note,
                    amp_envelope,
                    cutoff_envelope,
                    resonance_envelope,
                    ..
                }) if voice_id == Some(*candidate_voice_id)
                    || (channel == *candidate_channel && note == *candidate_note) =>
                {
                    context.send_event(NoteEvent::NoteOff {
                        timing: sample_offset,
                        voice_id: Some(*candidate_voice_id),
                        channel,
                        note,
                        velocity: 0.0,
                    });
    
                    // Trigger cutoff and resonance release envelopes
                    amp_envelope.release();
                    cutoff_envelope.release();
                    resonance_envelope.release();
    
                    *voice = None;
    
                    if voice_id.is_some() {
                        return;
                    }
                }
                _ => (),
            }
        }
    }
    
    
    fn waveform(&self) -> Waveform {
        self.params.waveform.value()
    }
}

const fn compute_fallback_voice_id(note: u8, channel: u8) -> i32 {
    note as i32 | ((channel as i32) << 16)
}

impl ClapPlugin for SubSynth {
    const CLAP_ID: &'static str = "art.taellinglin";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("A Polyphonic Subtractive Synthesizer");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::Instrument,
        ClapFeature::Synthesizer,
        ClapFeature::Stereo,
    ];

    const CLAP_POLY_MODULATION_CONFIG: Option<PolyModulationConfig> = Some(PolyModulationConfig {
        max_voice_capacity: NUM_VOICES,
        supports_overlapping_voices: true,
    });
}

impl Vst3Plugin for SubSynth {
    const VST3_CLASS_ID: [u8; 16] = *b"SubSynthLing0Lin";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Instrument,
        Vst3SubCategory::Synth,
        Vst3SubCategory::Stereo,
    ];
}


nih_export_clap!(SubSynth);
nih_export_vst3!(SubSynth);
