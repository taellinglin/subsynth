mod enums {
    use nih_plug::prelude::Enum;
    use crate::filter::FilterType;

    impl Enum for FilterType {
        fn variants() -> &'static [Self] {
            &[FilterType::Lowpass, FilterType::Bandpass, FilterType::Highpass, FilterType::Notch, FilterType::SVF]
        }
    }
}