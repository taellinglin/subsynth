//use atomic_float::AtomicF32;
use nih_plug::prelude::{Editor, GuiContext};
use nih_plug_iced::*;
use nih_plug_iced::widgets as nih_widgets;
use std::sync::Arc;
use nih_plug_iced::widget::{Text};
//use nih_plug_iced::Color;
use nih_plug_iced::widget::Space;
//use nih_plug_iced::Font;
use nih_plug_iced::Length;
//use nih_plug_iced::widget::*;
use crate::SubSynthParams;

// Remove impl TextStyle block

pub(crate) fn default_state() -> Arc<IcedState> {
    IcedState::from_size(400, 800)
}

pub(crate) fn create(
    params: Arc<SubSynthParams>,
    editor_state: Arc<IcedState>,
) -> Option<Box<dyn Editor>> {
    create_iced_editor::<SubSynthEditor>(editor_state, params)
}

struct SubSynthEditor {
    params: Arc<SubSynthParams>,
    context: Arc<dyn GuiContext>,

    gain_slider_state: nih_widgets::param_slider::State,
    waveform_slider_state: nih_widgets::param_slider::State,
    amp_attack_ms_slider_state: nih_widgets::param_slider::State,
    amp_release_ms_slider_state: nih_widgets::param_slider::State,
    amp_decay_ms_slider_state: nih_widgets::param_slider::State,
    amp_sustain_ms_slider_state: nih_widgets::param_slider::State,
    filter_cut_attack_ms_slider_state: nih_widgets::param_slider::State,
    filter_cut_decay_ms_slider_state: nih_widgets::param_slider::State,
    filter_cut_sustain_ms_slider_state: nih_widgets::param_slider::State,
    filter_cut_release_ms_slider_state: nih_widgets::param_slider::State,
    filter_res_attack_ms_slider_state: nih_widgets::param_slider::State,
    filter_res_decay_ms_slider_state: nih_widgets::param_slider::State,
    filter_res_sustain_ms_slider_state: nih_widgets::param_slider::State,
    filter_type_slider_state: nih_widgets::param_slider::State,
    filter_cut_slider_state: nih_widgets::param_slider::State,
    filter_res_slider_state: nih_widgets::param_slider::State,
}


#[derive(Debug, Clone, Copy)]
enum Message {
    /// Update a parameter's value.
    ParamUpdate(nih_widgets::ParamMessage),
}

impl IcedEditor for SubSynthEditor {
    type Executor = executor::Default;
    type Message = Message;
    type InitializationFlags = Arc<SubSynthParams>;

    fn new(
        params: Self::InitializationFlags,
        context: Arc<dyn GuiContext>,
    ) -> (Self, Command<Self::Message>) {
        let editor = SubSynthEditor {
            params,
            context,
            gain_slider_state: Default::default(),
            waveform_slider_state: Default::default(),
            amp_attack_ms_slider_state: Default::default(),
            amp_release_ms_slider_state: Default::default(),
            amp_decay_ms_slider_state: Default::default(),
            amp_sustain_ms_slider_state: Default::default(),
            filter_cut_attack_ms_slider_state: Default::default(),
            filter_cut_decay_ms_slider_state: Default::default(),
            filter_cut_sustain_ms_slider_state: Default::default(),
            filter_cut_release_ms_slider_state: Default::default(),
            filter_res_attack_ms_slider_state: Default::default(),
            filter_res_decay_ms_slider_state: Default::default(),
            filter_res_sustain_ms_slider_state: Default::default(),
            filter_type_slider_state: Default::default(),
            filter_cut_slider_state: Default::default(),
            filter_res_slider_state: Default::default(),
        };
        
    
        (editor, Command::none())
    }
    

    fn context(&self) -> &dyn GuiContext {
        self.context.as_ref()
    }

    fn update(
        &mut self,
        _window: &mut WindowQueue,
        message: Self::Message,
    ) -> Command<Self::Message> {
        match message {
            Message::ParamUpdate(message) => self.handle_param_message(message),
        }

        Command::none()
    }

    fn view(&mut self) -> Element<'_, Self::Message> {
        Column::new()
            .align_items(Alignment::Center)
            .push(
                Text::new("SubSynth")
                    .size(40)
                    .height(Length::Units(40))
                    .width(Length::Fill)
                    .horizontal_alignment(alignment::Horizontal::Center)
                    .vertical_alignment(alignment::Vertical::Bottom),
            )
            .push(
                Text::new("Gain")
                    .height(Length::Units(20))
                    .width(Length::Fill)
                    .horizontal_alignment(alignment::Horizontal::Center)
                    .vertical_alignment(alignment::Vertical::Center),
            )
            .push(
                nih_widgets::ParamSlider::new(&mut self.gain_slider_state, &self.params.gain)
                    .map(Message::ParamUpdate),
            )
            .push(
                Text::new("Waveform")
                    .height(Length::Units(20))
                    .width(Length::Fill)
                    .horizontal_alignment(alignment::Horizontal::Center)
                    .vertical_alignment(alignment::Vertical::Center),
            )
            .push(
                nih_widgets::ParamSlider::new(&mut self.waveform_slider_state, &self.params.waveform)
                    .map(Message::ParamUpdate),
            )
            .push(
                Text::new("Attack")
                    .height(Length::Units(20))
                    .width(Length::Fill)
                    .horizontal_alignment(alignment::Horizontal::Center)
                    .vertical_alignment(alignment::Vertical::Center),
            )
            .push(
                nih_widgets::ParamSlider::new(&mut self.amp_attack_ms_slider_state, &self.params.amp_attack_ms)
                    .map(Message::ParamUpdate),
            )
            .push(
                Text::new("Decay")
                    .height(Length::Units(20))
                    .width(Length::Fill)
                    .horizontal_alignment(alignment::Horizontal::Center)
                    .vertical_alignment(alignment::Vertical::Center),
            )
            .push(
                nih_widgets::ParamSlider::new(&mut self.amp_decay_ms_slider_state, &self.params.amp_decay_ms)
                    .map(Message::ParamUpdate),
            )
            .push(
                Text::new("Sustain")
                    .height(Length::Units(20))
                    .width(Length::Fill)
                    .horizontal_alignment(alignment::Horizontal::Center)
                    .vertical_alignment(alignment::Vertical::Center),
            )
            .push(
                nih_widgets::ParamSlider::new(&mut self.amp_sustain_ms_slider_state, &self.params.amp_sustain_ms)
                    .map(Message::ParamUpdate),
            )
            .push(
                Text::new("Release")
                    .height(Length::Units(20))
                    .width(Length::Fill)
                    .horizontal_alignment(alignment::Horizontal::Center)
                    .vertical_alignment(alignment::Vertical::Center),
            )
            .push(
                nih_widgets::ParamSlider::new(&mut self.amp_release_ms_slider_state, &self.params.amp_release_ms)
                    .map(Message::ParamUpdate),
            )
            .push(
                Text::new("Filter Type")
                    .height(Length::Units(20))
                    .width(Length::Fill)
                    .horizontal_alignment(alignment::Horizontal::Center)
                    .vertical_alignment(alignment::Vertical::Center),
            )
            .push(
                nih_widgets::ParamSlider::new(&mut self.filter_type_slider_state, &self.params.filter_type)
                    .map(Message::ParamUpdate),
            )
            .push(
                Text::new("Filter Cut")
                    .height(Length::Units(20))
                    .width(Length::Fill)
                    .horizontal_alignment(alignment::Horizontal::Center)
                    .vertical_alignment(alignment::Vertical::Center),
            )
            .push(
                nih_widgets::ParamSlider::new(&mut self.filter_cut_slider_state, &self.params.filter_cut)
                    .map(Message::ParamUpdate),
            )
            .push(
                Text::new("Filter Res")
                    .height(Length::Units(20))
                    .width(Length::Fill)
                    .horizontal_alignment(alignment::Horizontal::Center)
                    .vertical_alignment(alignment::Vertical::Center),
            )
            .push(
                nih_widgets::ParamSlider::new(&mut self.filter_res_slider_state, &self.params.filter_res)
                    .map(Message::ParamUpdate),
            )
            
            .push(
                Text::new("Filter Cut Attack")
                    .height(Length::Units(20))
                    .width(Length::Fill)
                    .horizontal_alignment(alignment::Horizontal::Center)
                    .vertical_alignment(alignment::Vertical::Center),
            )
            .push(
                nih_widgets::ParamSlider::new(&mut self.filter_cut_attack_ms_slider_state, &self.params.filter_cut_attack_ms)
                    .map(Message::ParamUpdate),
            )
            .push(
                Text::new("Filter Cut Decay")
                    .height(Length::Units(20))
                    .width(Length::Fill)
                    .horizontal_alignment(alignment::Horizontal::Center)
                    .vertical_alignment(alignment::Vertical::Center),
            )
            .push(
                nih_widgets::ParamSlider::new(&mut self.filter_cut_decay_ms_slider_state, &self.params.filter_cut_decay_ms)
                    .map(Message::ParamUpdate),
            )
            .push(
                Text::new("Filter Cut Sustain")
                    .height(Length::Units(20))
                    .width(Length::Fill)
                    .horizontal_alignment(alignment::Horizontal::Center)
                    .vertical_alignment(alignment::Vertical::Center),
            )
            .push(
                nih_widgets::ParamSlider::new(&mut self.filter_cut_sustain_ms_slider_state, &self.params.filter_cut_sustain_ms)
                    .map(Message::ParamUpdate),
            )
            .push(
                Text::new("Filter Cut Release")
                    .height(Length::Units(20))
                    .width(Length::Fill)
                    .horizontal_alignment(alignment::Horizontal::Center)
                    .vertical_alignment(alignment::Vertical::Center),
            )
            .push(
                nih_widgets::ParamSlider::new(&mut self.filter_cut_release_ms_slider_state, &self.params.filter_cut_release_ms)
                    .map(Message::ParamUpdate),
            )
            .push(
                Text::new("Filter Resonance Attack")
                    .height(Length::Units(20))
                    .width(Length::Fill)
                    .horizontal_alignment(alignment::Horizontal::Center)
                    .vertical_alignment(alignment::Vertical::Center),
            )
            .push(
                nih_widgets::ParamSlider::new(&mut self.filter_res_attack_ms_slider_state, &self.params.filter_res_attack_ms)
                    .map(Message::ParamUpdate),
            )
            .push(
                Text::new("Filter Resonance Decay")
                    .height(Length::Units(20))
                    .width(Length::Fill)
                    .horizontal_alignment(alignment::Horizontal::Center)
                    .vertical_alignment(alignment::Vertical::Center),
            )
            .push(
                nih_widgets::ParamSlider::new(&mut self.filter_res_decay_ms_slider_state, &self.params.filter_res_decay_ms)
                    .map(Message::ParamUpdate),
            )
            .push(
                Text::new("Filter Resonance Sustain")
                    .height(Length::Units(20))
                    .width(Length::Fill)
                    .horizontal_alignment(alignment::Horizontal::Center)
                    .vertical_alignment(alignment::Vertical::Center),
            )
            .push(
                nih_widgets::ParamSlider::new(&mut self.filter_res_sustain_ms_slider_state, &self.params.filter_res_sustain_ms)
                    .map(Message::ParamUpdate),
            )
            .push(Space::new(Length::Units(0), Length::Units(10)))
            .into()
    }
    

    fn background_color(&self) -> nih_plug_iced::Color {
        nih_plug_iced::Color {
            r: 0.82,
            g: 0.82,
            b: 0.82,
            a: 1.0,
        }
    }
}
