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

        stage_position_widget = PositionWidget(ui, _("Stage"), instrument, "stage_position_m")

        beam_shift_widget = PositionWidget(ui, _("Beam"), instrument, "beam_shift_m")

        column = self.content_widget

        column.add(stage_position_widget)
        column.add(beam_shift_widget)
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
