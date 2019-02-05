# standard libraries
import gettext

# local libraries
from nion.swift import Panel
from nion.swift import Workspace
from nion.ui import Widgets
from nion.utils import Binding
from nion.utils import Converter
from nion.utils import Geometry

from . import InstrumentDevice

_ = gettext.gettext


class PositionWidget(Widgets.CompositeWidgetBase):

    def __init__(self, ui, label: str, object, xy_property):
        super().__init__(ui.create_row_widget())

        def update_value(p, attribute_name, value):
            args = {"x": p.x, "y": p.y}
            args[attribute_name] = value
            return Geometry.FloatPoint(**args)

        stage_x_field = ui.create_line_edit_widget()
        stage_x_field.bind_text(Binding.PropertyAttributeBinding(object, xy_property, "x", Converter.PhysicalValueToStringConverter("nm", 1E9), update_attribute_fn=update_value))

        stage_y_field = ui.create_line_edit_widget()
        stage_y_field.bind_text(Binding.PropertyAttributeBinding(object, xy_property, "y", Converter.PhysicalValueToStringConverter("nm", 1E9), update_attribute_fn=update_value))

        row = self.content_widget

        row.add_spacing(8)
        row.add(ui.create_label_widget(label))
        row.add_spacing(8)
        row.add(ui.create_label_widget(_("X")))
        row.add_spacing(8)
        row.add(stage_x_field)
        row.add_spacing(8)
        row.add(ui.create_label_widget(_("Y")))
        row.add_spacing(8)
        row.add(stage_y_field)
        row.add_spacing(8)


class InstrumentWidget(Widgets.CompositeWidgetBase):

    def __init__(self, document_controller, instrument: InstrumentDevice.Instrument):
        super().__init__(document_controller.ui.create_column_widget(properties={"margin": 6, "spacing": 2}))

        self.document_controller = document_controller

        ui = document_controller.ui

        voltage_field = ui.create_line_edit_widget()
        voltage_field.bind_text(Binding.PropertyBinding(instrument, "voltage", converter=Converter.PhysicalValueToStringConverter(units="keV", multiplier=1E-3)))

        beam_current_field = ui.create_line_edit_widget()
        beam_current_field.bind_text(Binding.PropertyBinding(instrument, "beam_current", converter=Converter.PhysicalValueToStringConverter(units="pA", multiplier=1E12)))

        stage_position_widget = PositionWidget(ui, _("Stage"), instrument, "stage_position_m")

        beam_shift_widget = PositionWidget(ui, _("Beam"), instrument, "beam_shift_m")

        defocus_field = ui.create_line_edit_widget()
        defocus_field.bind_text(Binding.PropertyBinding(instrument, "C10", converter=Converter.PhysicalValueToStringConverter(units="nm", multiplier=1E9)))

        c12_widget = PositionWidget(ui, _("C12"), instrument, "C12")

        c21_widget = PositionWidget(ui, _("C21"), instrument, "C21")

        c23_widget = PositionWidget(ui, _("C23"), instrument, "C23")

        c3_field = ui.create_line_edit_widget()
        c3_field.bind_text(Binding.PropertyBinding(instrument, "C30", converter=Converter.PhysicalValueToStringConverter(units="nm", multiplier=1E9)))

        c32_widget = PositionWidget(ui, _("C32"), instrument, "C32")

        c34_widget = PositionWidget(ui, _("C34"), instrument, "C34")

        blanked_checkbox = ui.create_check_box_widget(_("Beam Blanked"))
        blanked_checkbox.bind_checked(Binding.PropertyBinding(instrument, "is_blanked"))

        slit_in_checkbox = ui.create_check_box_widget(_("Slit In"))
        slit_in_checkbox.bind_checked(Binding.PropertyBinding(instrument, "is_slit_in"))

        energy_offset_field = ui.create_line_edit_widget()
        energy_offset_field.bind_text(Binding.PropertyBinding(instrument, "energy_offset_eV", converter=Converter.FloatToStringConverter()))

        energy_dispersion_field = ui.create_line_edit_widget()
        energy_dispersion_field.bind_text(Binding.PropertyBinding(instrument, "energy_per_channel_eV", converter=Converter.FloatToStringConverter()))

        beam_row = ui.create_row_widget()
        beam_row.add_spacing(8)
        beam_row.add(blanked_checkbox)
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

        column = self.content_widget

        column.add(voltage_row)
        column.add(beam_current_row)
        column.add(stage_position_widget)
        column.add(beam_shift_widget)
        column.add(defocus_row)
        column.add(c12_widget)
        column.add(c21_widget)
        column.add(c23_widget)
        column.add(c3_row)
        column.add(c32_widget)
        column.add(c34_widget)
        column.add(beam_row)
        column.add(eels_row)
        column.add_stretch()


class InstrumentControlPanel(Panel.Panel):

    def __init__(self, document_controller, panel_id, properties):
        super().__init__(document_controller, panel_id, panel_id)
        ui = document_controller.ui
        self.widget = ui.create_column_widget()
        instrument = properties["instrument"]
        instrument_widget = InstrumentWidget(self.document_controller, instrument)
        self.widget.add(instrument_widget)
        self.widget.add_spacing(12)
        self.widget.add_stretch()


def run(instrument: InstrumentDevice.Instrument) -> None:
    panel_id = "simulator-instrument-panel"
    name = _("Simulator Instrument")
    Workspace.WorkspaceManager().register_panel(InstrumentControlPanel, panel_id, name, ["left", "right"], "left", {"instrument": instrument})
