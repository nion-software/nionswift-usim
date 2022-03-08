from __future__ import annotations

# standard libraries
import functools
import gettext
import typing

# local libraries
from nion.swift import Panel
from nion.swift import Workspace
from nion.ui import Widgets
from nion.utils import Binding
from nion.utils import Converter
from nion.utils.ReferenceCounting import weak_partial

if typing.TYPE_CHECKING:
    from . import InstrumentDevice
    from nion.swift import DocumentController
    from nion.ui import UserInterface
    from nion.utils import Validator

_ = gettext.gettext


class Control2DBinding(Binding.Binding):
    def __init__(self, instrument: InstrumentDevice.Instrument, control_name: str, attribute_name: str,
                 converter: typing.Optional[Converter.ConverterLike[typing.Any, typing.Any]] = None,
                 fallback: typing.Any = None) -> None:
        super().__init__(instrument, converter=converter, fallback=fallback)

        def set_property_value(source: typing.Any, value: typing.Any) -> None:
            if source:
                getattr(instrument.get_control(control_name), attribute_name).set_output_value(value)

        def get_property_value(source: typing.Any) -> typing.Any:
            return getattr(instrument.get_control(control_name), attribute_name).output_value if source else None

        self.source_setter = weak_partial(set_property_value, self.source)
        self.source_getter = weak_partial(get_property_value, self.source)

        # thread safe
        def property_changed(property_name_: str) -> None:
            if property_name_ == control_name:
                # perform on the main thread
                value = self.source_getter() if callable(self.source_getter) else 0.0
                if value is not None:
                    self.update_target(value)
                else:
                    self.update_target_direct(self.fallback)

        self.__property_changed_listener = instrument.property_changed_event.listen(property_changed)

    def close(self) -> None:
        self.__property_changed_listener.close()
        self.__property_changed_listener = typing.cast(typing.Any, None)
        super().close()


class ControlBinding(Binding.Binding):
    def __init__(self, instrument: InstrumentDevice.Instrument, control_name: str, *,
                 converter: typing.Optional[Converter.ConverterLike[typing.Any, typing.Any]] = None,
                 validator: typing.Optional[Validator.ValidatorLike[typing.Any]] = None,
                 fallback: typing.Any = None) -> None:
        super().__init__(instrument, converter=converter, validator=validator, fallback=fallback)

        def set_property_value(source: typing.Any, value: typing.Any) -> None:
            if source:
                instrument.SetVal(control_name, value)

        def get_property_value(source: typing.Any) -> typing.Any:
            return instrument.GetVal(control_name) if source else None

        self.source_setter = weak_partial(set_property_value, self.source)
        self.source_getter = weak_partial(get_property_value, self.source)

        # thread safe
        def property_changed(property_name_: str) -> None:
            if property_name_ == control_name:
                value = self.source_getter() if callable(self.source_getter) else 0.0
                if value is not None:
                    self.update_target(value)
                else:
                    self.update_target_direct(self.fallback)

        self.__property_changed_listener = instrument.property_changed_event.listen(property_changed)

    def close(self) -> None:
        self.__property_changed_listener.close()
        self.__property_changed_listener = typing.cast(typing.Any, None)
        super().close()


class PositionWidget(Widgets.CompositeWidgetBase):

    def __init__(self, ui: UserInterface.UserInterface, label: str, instrument: InstrumentDevice.Instrument,
                 xy_property: str, unit: str = "nm", multiplier: float = 1E9) -> None:
        row_widget = ui.create_row_widget()
        super().__init__(row_widget)

        stage_x_field = ui.create_line_edit_widget()
        stage_x_field.bind_text(Control2DBinding(instrument, xy_property, "x", Converter.PhysicalValueToStringConverter(unit, multiplier)))

        stage_y_field = ui.create_line_edit_widget()
        stage_y_field.bind_text(Control2DBinding(instrument, xy_property, "y", Converter.PhysicalValueToStringConverter(unit, multiplier)))

        row_widget.add_spacing(8)
        row_widget.add(ui.create_label_widget(label))
        row_widget.add_spacing(8)
        row_widget.add(ui.create_label_widget(_("X")))
        row_widget.add_spacing(8)
        row_widget.add(stage_x_field)
        row_widget.add_spacing(8)
        row_widget.add(ui.create_label_widget(_("Y")))
        row_widget.add_spacing(8)
        row_widget.add(stage_y_field)
        row_widget.add_spacing(8)


class InstrumentWidget(Widgets.CompositeWidgetBase):

    def __init__(self, ui: UserInterface.UserInterface, instrument: InstrumentDevice.Instrument) -> None:
        column_widget = ui.create_column_widget(properties={"margin": 6, "spacing": 2})
        super().__init__(column_widget)

        sample_combo_box = ui.create_combo_box_widget(instrument.sample_titles)
        sample_combo_box.current_index = instrument.sample_index
        sample_combo_box.bind_current_index(Binding.PropertyBinding(instrument, "sample_index"))

        voltage_field = ui.create_line_edit_widget()
        voltage_field.bind_text(Binding.PropertyBinding(instrument, "voltage", converter=Converter.PhysicalValueToStringConverter(units="keV", multiplier=1E-3)))

        beam_current_field = ui.create_line_edit_widget()
        beam_current_field.bind_text(ControlBinding(instrument, "BeamCurrent", converter=Converter.PhysicalValueToStringConverter(units="pA", multiplier=1E12)))

        stage_position_widget = PositionWidget(ui, _("Stage"), instrument, "stage_position_m")

        beam_shift_widget = PositionWidget(ui, _("Beam"), instrument, "beam_shift_m")

        defocus_field = ui.create_line_edit_widget()
        defocus_field.bind_text(ControlBinding(instrument, "C10", converter=Converter.PhysicalValueToStringConverter(units="nm", multiplier=1E9)))

        c12_widget = PositionWidget(ui, _("C12"), instrument, "C12")

        c21_widget = PositionWidget(ui, _("C21"), instrument, "C21")

        c23_widget = PositionWidget(ui, _("C23"), instrument, "C23")

        c3_field = ui.create_line_edit_widget()
        c3_field.bind_text(ControlBinding(instrument, "C30", converter=Converter.PhysicalValueToStringConverter(units="nm", multiplier=1E9)))

        c32_widget = PositionWidget(ui, _("C32"), instrument, "C32")

        c34_widget = PositionWidget(ui, _("C34"), instrument, "C34")

        blanked_checkbox = ui.create_check_box_widget(_("Beam Blanked"))
        blanked_checkbox.bind_checked(Binding.PropertyBinding(instrument, "is_blanked"))

        slit_in_checkbox = ui.create_check_box_widget(_("Slit In"))
        slit_in_checkbox.bind_checked(Binding.PropertyBinding(instrument, "is_slit_in"))

        voa_in_checkbox = ui.create_check_box_widget(_("VOA In"))
        voa_in_checkbox.bind_checked(ControlBinding(instrument, "S_VOA"))

        convergenve_angle_field = ui.create_line_edit_widget()
        convergenve_angle_field.bind_text(ControlBinding(instrument, "ConvergenceAngle", converter=Converter.PhysicalValueToStringConverter(units="mrad", multiplier=1E3)))

        c_aperture_widget = PositionWidget(ui, _("CAperture"), instrument, "CAperture", unit="mrad", multiplier=1E3)
        aperture_round_widget = PositionWidget(ui, _("ApertureRound"), instrument, "ApertureRound", unit="", multiplier=1)

        energy_offset_field = ui.create_line_edit_widget()
        energy_offset_field.bind_text(Binding.PropertyBinding(instrument, "energy_offset_eV", converter=Converter.FloatToStringConverter()))

        energy_dispersion_field = ui.create_line_edit_widget()
        energy_dispersion_field.bind_text(Binding.PropertyBinding(instrument, "energy_per_channel_eV", converter=Converter.FloatToStringConverter()))

        beam_row = ui.create_row_widget()
        beam_row.add_spacing(8)
        beam_row.add(blanked_checkbox)
        beam_row.add_spacing(8)
        beam_row.add(voa_in_checkbox)
        beam_row.add_stretch()

        eels_row = ui.create_row_widget()
        eels_row.add_spacing(8)
        eels_row.add(slit_in_checkbox)
        eels_row.add_spacing(8)
        eels_row.add(ui.create_label_widget("+eV"))
        eels_row.add_spacing(4)
        eels_row.add(energy_offset_field)
        eels_row.add_spacing(8)
        eels_row.add(ui.create_label_widget("eV/ch"))
        eels_row.add_spacing(4)
        eels_row.add(energy_dispersion_field)
        eels_row.add_stretch()

        defocus_row = ui.create_row_widget()
        defocus_row.add_spacing(8)
        defocus_row.add_spacing(8)
        defocus_row.add(ui.create_label_widget("Defocus"))
        defocus_row.add(defocus_field)
        defocus_row.add_stretch()

        c3_row = ui.create_row_widget()
        c3_row.add_spacing(8)
        c3_row.add_spacing(8)
        c3_row.add(ui.create_label_widget("Spherical Aberration"))
        c3_row.add(c3_field)
        c3_row.add_stretch()

        sample_row = ui.create_row_widget()
        sample_row.add_spacing(8)
        sample_row.add_spacing(8)
        sample_row.add(sample_combo_box)
        sample_row.add_stretch()

        voltage_row = ui.create_row_widget()
        voltage_row.add_spacing(8)
        voltage_row.add_spacing(8)
        voltage_row.add(ui.create_label_widget("Voltage"))
        voltage_row.add(voltage_field)
        voltage_row.add_stretch()

        beam_current_row = ui.create_row_widget()
        beam_current_row.add_spacing(8)
        beam_current_row.add_spacing(8)
        beam_current_row.add(ui.create_label_widget("Beam Current"))
        beam_current_row.add(beam_current_field)
        beam_current_row.add_stretch()

        convergence_angle_row = ui.create_row_widget()
        convergence_angle_row.add_spacing(8)
        convergence_angle_row.add_spacing(8)
        convergence_angle_row.add(ui.create_label_widget("Convergence Angle"))
        convergence_angle_row.add(convergenve_angle_field)
        convergence_angle_row.add_stretch()

        column_widget.add(sample_row)
        column_widget.add(voltage_row)
        column_widget.add(beam_current_row)
        column_widget.add(stage_position_widget)
        column_widget.add(beam_shift_widget)
        column_widget.add(defocus_row)
        column_widget.add(c12_widget)
        column_widget.add(c21_widget)
        column_widget.add(c23_widget)
        column_widget.add(c3_row)
        column_widget.add(c32_widget)
        column_widget.add(c34_widget)
        column_widget.add(beam_row)
        column_widget.add(convergence_angle_row)
        column_widget.add(c_aperture_widget)
        column_widget.add(aperture_round_widget)
        column_widget.add(eels_row)
        column_widget.add_stretch()


class InstrumentControlPanel(Panel.Panel):

    def __init__(self, document_controller: DocumentController.DocumentController, panel_id: str, properties: typing.Mapping[str, typing.Any]) -> None:
        super().__init__(document_controller, panel_id, panel_id)
        ui = document_controller.ui
        self.widget = ui.create_column_widget()
        instrument = properties["instrument"]
        instrument_widget = InstrumentWidget(ui, instrument)
        self.widget.add(instrument_widget)
        self.widget.add_spacing(12)
        self.widget.add_stretch()


def run(instrument: InstrumentDevice.Instrument) -> None:
    panel_id = "simulator-instrument-panel"
    name = _("Simulator Instrument")
    Workspace.WorkspaceManager().register_panel(InstrumentControlPanel, panel_id, name, ["left", "right"], "left", {"instrument": instrument})
