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
    #[id = "gain"]
    gain: FloatParam,
    #[id = "amp_atk"]
    amp_attack_ms: FloatParam,
    #[id = "amp_rel"]
    amp_release_ms: FloatParam,
}

#[derive(Debug, Clone)]
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
            voices: [0; NUM_VOICES as usize].map(|_| None),
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
        }
    }
}

impl Plugin for PolyModSynth {
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

    fn reset(&mut self) {
        self.prng = Pcg32::new(420, 1337);

        self.voices.fill(None);
        self.next_internal_voice_id = 0;
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let num_samples = buffer.samples();
        let sample_rate = context.transport().sample_rate;
        let output = buffer.as_slice();

        let mut next_event = context.next_event();
        let mut block_start: usize = 0;
        let mut block_end: usize = MAX_BLOCK_SIZE.min(num_samples);
        while block_start < num_samples {
            let this_sample_internal_voice_id_start = self.next_internal_voice_id;
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
                                let amp_envelope = Smoother::new(SmoothingStyle::Exponential(
                                    self.params.amp_attack_ms.value(),
                                ));
                                amp_envelope.reset(0.0);
                                amp_envelope.set_target(sample_rate, 1.0);

                                let voice =
                                    self.start_voice(context, timing, voice_id, channel, note);
                                voice.velocity_sqrt = velocity.sqrt();
                                voice.phase = initial_phase;
                                voice.phase_delta = util::midi_note_to_freq(note) / sample_rate;
                                voice.amp_envelope = amp_envelope;
                            }
                            NoteEvent::NoteOff {
                                timing: _,
                                voice_id,
                                channel,
                                note,
                                velocity: _,
                            } => {
                                self.start_release_for_voices(sample_rate, voice_id, channel, note)
                            }
                            NoteEvent::Choke {
                                timing,
                                voice_id,
                                channel,
                                note,
                            } => {
                                self.choke_voices(context, timing, voice_id, channel, note);
                            }
                            NoteEvent::PolyModulation {
                                timing: _,
                                voice_id,
                                poly_modulation_id,
                                normalized_offset,
                            } => {
                                if let Some(voice_idx) = self.get_voice_idx(voice_id) {
                                    let voice = self.voices[voice_idx].as_mut().unwrap();

                                    match poly_modulation_id {
                                        GAIN_POLY_MOD_ID => {
                                            let target_plain_value = self
                                                .params
                                                .gain
                                                .preview_modulated(normalized_offset);
                                            let (_, smoother) =
                                                voice.voice_gain.get_or_insert_with(|| {
                                                    (
                                                        normalized_offset,
                                                        self.params.gain.smoothed.clone(),
                                                    )
                                                });
                                            if voice.internal_voice_id
                                                >= this_sample_internal_voice_id_start
                                            {
                                                smoother.reset(target_plain_value);
                                            } else {
                                                smoother
                                                    .set_target(sample_rate, target_plain_value);
                                            }
                                        }
                                        n => nih_debug_assert_failure!(
                                            "Polyphonic modulation sent for unknown poly \
                                             modulation ID {}",
                                            n
                                        ),
                                    }
                                }
                            }
                            NoteEvent::MonoAutomation {
                                timing: _,
                                poly_modulation_id,
                                normalized_value,
                            } => {
                                for voice in self.voices.iter_mut().filter_map(|v| v.as_mut()) {
                                    match poly_modulation_id {
                                        GAIN_POLY_MOD_ID => {
                                            let (normalized_offset, smoother) =
                                                match voice.voice_gain.as_mut() {
                                                    Some((o, s)) => (o, s),
                                                    None => continue,
                                                };
                                            let target_plain_value =
                                                self.params.gain.preview_plain(
                                                    normalized_value + *normalized_offset,
                                                );
                                            smoother.set_target(sample_rate, target_plain_value);
                                        }
                                        n => nih_debug_assert_failure!(
                                            "Automation event sent for unknown poly modulation ID \
                                             {}",
                                            n
                                        ),
                                    }
                                }
                            }
                            _ => (),
                        };

                        next_event = context.next_event();
                    }
                    Some(event) if (event.timing() as usize) < block_end => {
                        block_end = event.timing() as usize;
                        break 'events;
                    }
                    _ => break 'events,
                }
            }
            output[0][block_start..block_end].fill(0.0);
            output[1][block_start..block_end].fill(0.0);
            let block_len = block_end - block_start;
            let mut gain = [0.0; MAX_BLOCK_SIZE];
            let mut voice_gain = [0.0; MAX_BLOCK_SIZE];
            let mut voice_amp_envelope = [0.0; MAX_BLOCK_SIZE];
            self.params.gain.smoothed.next_block(&mut gain, block_len);

            // TODO: Some form of band limiting
            // TODO: Filter
            for voice in self.voices.iter_mut().filter_map(|v| v.as_mut()) {
                let gain = match &voice.voice_gain {
                    Some((_, smoother)) => {
                        smoother.next_block(&mut voice_gain, block_len);
                        &voice_gain
                    }
                    None => &gain,
                };
                voice
                    .amp_envelope
                    .next_block(&mut voice_amp_envelope, block_len);

                for (value_idx, sample_idx) in (block_start..block_end).enumerate() {
                    let amp = voice.velocity_sqrt * gain[value_idx] * voice_amp_envelope[value_idx];
                    let sample = triangle_wave(voice.phase) * amp;

                    voice.phase += voice.phase_delta;
                    if voice.phase >= 1.0 {
                        voice.phase -= 1.0;
                    }

                    output[0][sample_idx] += sample;
                    output[1][sample_idx] += sample;
                }
            }
            for voice in self.voices.iter_mut() {
                match voice {
                    Some(v) if v.releasing && v.amp_envelope.previous_value() == 0.0 => {
                        context.send_event(NoteEvent::VoiceTerminated {
                            timing: block_end as u32,
                            voice_id: Some(v.voice_id),
                            channel: v.channel,
                            note: v.note,
                        });
                        *voice = None;
                    }
                    _ => (),
                }
            }
            block_start = block_end;
            block_end = (block_start + MAX_BLOCK_SIZE).min(num_samples);
        }

        ProcessStatus::Normal
    }
}

impl PolyModSynth {
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
        let new_voice = Voice {
            voice_id: voice_id.unwrap_or_else(|| compute_fallback_voice_id(note, channel)),
            internal_voice_id: self.next_internal_voice_id,
            channel,
            note,
            velocity_sqrt: 1.0,

            phase: 0.0,
            phase_delta: 0.0,
            releasing: false,
            amp_envelope: Smoother::none(),

            voice_gain: None,
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
                    context.send_event(NoteEvent::VoiceTerminated {
                        timing: sample_offset,
                        voice_id: Some(oldest_voice.voice_id),
                        channel: oldest_voice.channel,
                        note: oldest_voice.note,
                    });
                }

                *oldest_voice = Some(new_voice);
                return oldest_voice.as_mut().unwrap();
            }
        }
    }

    fn start_release_for_voices(
        &mut self,
        sample_rate: f32,
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
                    ..
                }) if voice_id == Some(*candidate_voice_id)
                    || (channel == *candidate_channel && note == *candidate_note) =>
                {
                    *releasing = true;
                    amp_envelope.style =
                        SmoothingStyle::Exponential(self.params.amp_release_ms.value());
                    amp_envelope.set_target(sample_rate, 0.0);
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
                    ..
                }) if voice_id == Some(*candidate_voice_id)
                    || (channel == *candidate_channel && note == *candidate_note) =>
                {
                    context.send_event(NoteEvent::VoiceTerminated {
                        timing: sample_offset,
                        voice_id: Some(*candidate_voice_id),
                        channel,
                        note,
                    });
                    *voice = None;

                    if voice_id.is_some() {
                        return;
                    }
                }
                _ => (),
            }
        }
    }
}

const fn compute_fallback_voice_id(note: u8, channel: u8) -> i32 {
    note as i32 | ((channel as i32) << 16)
}

impl ClapPlugin for PolyModSynth {
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

impl Vst3Plugin for PolyModSynth {
    const VST3_CLASS_ID: [u8; 16] = *b"PolyM0dSynth1337";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Instrument,
        Vst3SubCategory::Synth,
        Vst3SubCategory::Stereo,
    ];
}

fn triangle_wave(phase: f32) -> f32 {
    if phase <= 0.5 {
        phase * 4.0 - 1.0
    } else {
        (1.0 - phase) * 4.0 - 1.0
    }
}

nih_export_clap!(PolyModSynth);
nih_export_vst3!(PolyModSynth);
