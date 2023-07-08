use nih_plug::param::*;
use nih_plug::util;

#[derive(Params)]
pub struct SubSynthParams {
    pub gain: FloatParam,
    pub waveform: FloatParam,
    pub amp_attack_ms: FloatParam,
    pub amp_release_ms: FloatParam,
    pub amp_decay_ms: FloatParam,
    pub amp_sustain_ms: FloatParam,
    pub filter_cut_attack_ms: FloatParam,
    pub filter_cut_decay_ms: FloatParam,
    pub filter_cut_sustain_ms: FloatParam,
    pub filter_cut_release_ms: FloatParam,
    pub filter_res_attack_ms: FloatParam,
    pub filter_res_decay_ms: FloatParam,
    pub filter_res_sustain_ms: FloatParam,
    pub filter_type: FloatParam,
    pub filter_cut: FloatParam,
    pub filter_res: FloatParam,
}

impl Default for SubSynthParams {
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
            .with_smoother(SmoothingStyle::Logarithmic(5.0))
            .with_unit(" dB"),
            waveform: FloatParam::new(
                "Waveform",
                0.0,
                FloatRange::Linear {
                    min: 0.0,
                    max: 1.0,
                },
            ),
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
            amp_sustain_ms: FloatParam::new(
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
            filter_cut_attack_ms: FloatParam::new(
                "Filter Cut Attack",
                100.0,
                FloatRange::Linear {
                    min: 0.0,
                    max: 1000.0,
                },
            )
            .with_step_size(0.1)
            .with_unit(" ms"),
            filter_cut_decay_ms: FloatParam::new(
                "Filter Cut Decay",
                100.0,
                FloatRange::Linear {
                    min: 0.0,
                    max: 1000.0,
                },
            )
            .with_step_size(0.1)
            .with_unit(" ms"),
            filter_cut_sustain_ms: FloatParam::new(
                "Filter Cut Sustain",
                1000.0,
                FloatRange::Linear {
                    min: 0.0,
                    max: 5000.0,
                },
            )
            .with_step_size(0.1)
            .with_unit(" ms"),
            filter_cut_release_ms: FloatParam::new(
                "Filter Cut Release",
                100.0,
                FloatRange::Linear {
                    min: 0.0,
                    max: 1000.0,
                },
            )
            .with_step_size(0.1)
            .with_unit(" ms"),
            filter_res_attack_ms: FloatParam::new(
                "Filter Resonance Attack",
                100.0,
                FloatRange::Linear {
                    min: 0.0,
                    max: 1000.0,
                },
            )
            .with_step_size(0.1)
            .with_unit(" ms"),
            filter_res_decay_ms: FloatParam::new(
                "Filter Resonance Decay",
                100.0,
                FloatRange::Linear {
                    min: 0.0,
                    max: 1000.0,
                },
            )
            .with_step_size(0.1)
            .with_unit(" ms"),
            filter_res_sustain_ms: FloatParam::new(
                "Filter Resonance Sustain",
                1000.0,
                FloatRange::Linear {
                    min: 0.0,
                    max: 5000.0,
                },
            )
            .with_step_size(0.1)
            .with_unit(" ms"),
            filter_type: FloatParam::new(
                "Filter Type",
                0.0,
                FloatRange::Linear {
                    min: 0.0,
                    max: 1.0,
                },
            ),
            filter_cut: FloatParam::new(
                "Filter Cut",
                0.0,
                FloatRange::Linear {
                    min: 0.0,
                    max: 1.0,
                },
            ),
            filter_res: FloatParam::new(
                "Filter Res",
                0.0,
                FloatRange::Linear {
                    min: 0.0,
                    max: 1.0,
                },
            ),
        }
    }
}
