### ui_main.py
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QScrollArea, QPushButton,
    QCheckBox, QSlider, QHBoxLayout, QFrame, QLabel, QListWidget, QSizePolicy,
    QListWidgetItem, QAbstractItemView
)
from PyQt5.QtCore import Qt
from frontend.plots import initialize_plot, update_plot, create_canvas, update_angle, get_arc_offset, create_radar_graph
from backend.data_manager import init_data, Basis, init_angle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from matplotlib.pyplot import close
import numpy as np


plt.rcParams['animation.embed_limit'] = 100.0

class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_view = "radar_graph"
        # self.current_view = "standard_view"  # Default view
        self.initUI()

    def initUI(self):
        self.setWindowTitle('LLL Visualization')
        self.setGeometry(100, 100, 1600, 900)  # Increased height for better spacing

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.setCentralWidget(self.scroll_area)

        self.container = QWidget()
        self.container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.scroll_area.setWidget(self.container)

        self.layout = QVBoxLayout(self.container)
        self.container.setLayout(self.layout)

        self.Basis = Basis
        self.repeat = True

        self.init_data()
        self.init_plot()
        self.setup_ui()
        self.animation()

    def init_data(self):
        self.data = init_data()
        # print(self.data)

    def init_plot(self):
        self.fig = Figure(figsize=(16, 12))  # Increased height for better spacing
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)

        if self.current_view == "standard_view":
            
            self.ax_top, self.ax_angles, self.Basis_interp_line, self.target_interp_line, self.x_interp_line = initialize_plot(
                                                                                                self.fig, self.data['vec_dimension'], 
                                                                                                self.Basis, self.data['scale'], self.data['method'],
                                                                                                self.Basis, self.data['target_Basis']
                                                                                            )
            self.update_plot()

        elif self.current_view == "radar_graph":
            # Prepare data for the radar graph
            self.radar_data = create_radar_graph(
                                self.fig,
                                np.abs(self.Basis), 
                                np.abs(self.data['target_Basis']),
                                labels = [f"Dimension {i + 1}" for i in range(self.Basis.shape[1])]
                            )
            self.radar_data_start = self.radar_data['data_start']
            self.radar_data_end = self.radar_data['data_end']
            self.radar_lines = self.radar_data['lines']
            self.radar_fills = self.radar_data['fills']
            self.radar_ax = self.radar_data['ax']
            self.radar_theta = self.radar_data['theta']

            max_val = max(np.max(self.radar_data_start), np.max(self.radar_data_end))
            self.radar_ax.set_ylim(0, max_val + 10)
        
    
    def update_plot(self, offsets = []):
        self.scatter, self.line_ori, self.line_interp, self.ax_angles, self.angle_plots = update_plot(
            self.ax_top, 
            self.data['vec_dimension'], 
            self.x_interp_line, 
            self.Basis_interp_line,

            self.ax_angles,
            self.data['angles_basis_complementary'],
            self.data['target_angles_basis_complementary'],
            self.fig
        )
    
    def setup_ui(self):
        menubar = self.menuBar()        
        self.setup_ui_main()
        self.setup_menubar(menubar)
    
    def setup_ui_main(self):
        # Create a frame for controls
        self.controls_frame = QFrame(self)
        self.controls_layout = QHBoxLayout(self.controls_frame)

        # Play/Pause Button
        self.play_button = QPushButton('Pause', self)
        self.play_button.clicked.connect(self.on_toggle)

        # Loop Checkbox
        self.loop_checkbox = QCheckBox('Loop', self)
        self.loop_checkbox.setChecked(True)
        self.loop_checkbox.stateChanged.connect(self.on_checkbox_toggle)

        # Slider for animation frames
        self.slider_bar = QSlider(Qt.Horizontal, self)
        self.slider_bar.setMinimum(0)
        self.slider_bar.setMaximum(self.data['total_frames'] - 1)
        self.slider_bar.setValue(0)
        self.slider_bar.valueChanged.connect(self.on_slider)

        # Add widgets to controls layout
        self.controls_layout.addWidget(self.play_button)
        self.controls_layout.addWidget(self.loop_checkbox)
        self.controls_layout.addWidget(QLabel("Frame:"))
        self.controls_layout.addWidget(self.slider_bar)

        self.layout.addWidget(self.controls_frame)

        # Create a frame for selection controls
        self.selection_frame = QFrame(self)
        self.selection_layout = QVBoxLayout(self.selection_frame)

        # Label for selection
        self.selection_label = QLabel("Select Basis Vectors to Display:")
        self.selection_layout.addWidget(self.selection_label)

        # List widget for basis vector selection
        self.basis_list = QListWidget(self)
        self.basis_list.setSelectionMode(QAbstractItemView.MultiSelection)
        for i in range(self.Basis.shape[0]):
            item = QListWidgetItem(f'Basis Vector {i + 1}')
            item.setSelected(True)  # Default to selected
            self.basis_list.addItem(item)
        self.basis_list.itemSelectionChanged.connect(self.update_selected_arcs)
        self.selection_layout.addWidget(self.basis_list)

        self.layout.addWidget(self.selection_frame)

        # Add Navigation Toolbar for zoom and pan
        from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout.addWidget(self.toolbar)
    
    def setup_menubar(self,menubar):
        self.setup_toggle_menu(menubar)
        self.setup_view_menu(menubar)

    def setup_toggle_menu(self,menubar):
        toggle_menu = menubar.addMenu('Toggle Visibility')

        scatter_action = toggle_menu.addAction('Toggle All Scatters')
        scatter_action.setCheckable(True)
        scatter_action.setChecked(True)
        scatter_action.triggered.connect(lambda checked: self.toggle_all_visibility('scatter', checked))

        line_ori_action = toggle_menu.addAction('Toggle All Basis')
        line_ori_action.setCheckable(True)
        line_ori_action.setChecked(True)
        line_ori_action.triggered.connect(lambda checked: self.toggle_all_visibility('line_ori', checked))

        line_interp_action = toggle_menu.addAction('Toggle All Interpolations')
        line_interp_action.setCheckable(True)
        line_interp_action.setChecked(True)
        line_interp_action.triggered.connect(lambda checked: self.toggle_all_visibility('line_interp', checked))

    def toggle_all_visibility(self, element, checked):
        ''' Toggle visibility of different plot elements '''
        if element == 'scatter':
            for scatter in self.scatter:
                scatter.set_visible(checked)
        elif element == 'line_ori':
            for line in self.line_ori:
                line.set_visible(checked)
        elif element == 'line_interp':
            for line in self.line_interp:
                line.set_visible(checked)
        self.canvas.draw_idle()

    def setup_view_menu(self, menubar):
        # Create "View" menu
        view_menu = menubar.addMenu('View')

        # Add actions for switching views
        radar_view_action = view_menu.addAction('Radar Graph View')
        radar_view_action.triggered.connect(self.show_radar_graph)

        standard_view_action = view_menu.addAction('Standard View')
        standard_view_action.triggered.connect(self.show_standard_view)

    def show_radar_graph(self):
        self.current_view = "radar_graph"
        for i in reversed(range(self.layout.count())):
            widget = self.layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        # Clear the old figure, if any
        if hasattr(self, 'current_fig'):
            self.clear_figure(self.current_fig)

        # print(f"Radar Lines: {self.radar_lines}")
        # print(f"Radar Fills: {self.radar_fills}")
        self.init_plot()
        self.canvas.draw_idle()
        self.setup_ui_main()

    def show_standard_view(self):
        self.current_view = "standard_view"

        # Clear existing layout
        for i in reversed(range(self.layout.count())):
            widget = self.layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        
        # Clear the old figure, if any
        if hasattr(self, 'current_fig'):
            self.clear_figure(self.current_fig)

        # Reinitialize the standard visualization view
        self.init_plot()
        self.canvas.draw_idle()
        self.setup_ui_main()

    def clear_figure(self,fig):
        """
        Clears and closes the given figure to prevent residual elements.
        """
        if fig:
            fig.clf()  # Clear the figure
            close(fig)  # Close the figure to free memory

    def on_hover(self, event):
        ''' Display annotation when hovering over a scatter or line '''
        vis = False
        if event.inaxes == self.ax_top:
            for i, (scatter, line) in enumerate(zip(self.scatter, self.line_ori)):
                cont_scatter, _ = scatter.contains(event)
                cont_line, _ = line.contains(event)
                if cont_scatter or cont_line:
                    angle = self.angles_basis_complementary[i]
                    self.annotation.xy = (event.xdata, event.ydata)
                    self.annotation.set_text(f'Basis {i+1}\nAngle: {angle}°')
                    self.annotation.set_visible(True)
                    vis = True
                    break
        if not vis:
            self.annotation.set_visible(False)
        self.canvas.draw_idle()

    def update_selected_arcs(self):
        ''' Update which arcs and basis vectors are displayed based on selection '''
        selected_items = self.basis_list.selectedItems()
        selected_indices = [int(item.text().split(' ')[-1]) - 1 for item in selected_items]

        # Hide all arcs and basis vectors first
        for i, plot in enumerate(self.angle_plots):
            arc, line1, line2, angle_text, _, _ = plot
            arc.set_visible(False)
            line1.set_visible(False)
            line2.set_visible(False)
            angle_text.set_visible(False)
            # Also hide corresponding basis vectors
            self.scatter[i].set_visible(False)
            self.line_ori[i].set_visible(False)
            self.line_interp[i].set_visible(False)

        # Show only selected arcs and basis vectors
        for i in selected_indices:
            if i < len(self.angle_plots):
                arc, line1, line2, angle_text, _, _ = self.angle_plots[i]
                arc.set_visible(True)
                line1.set_visible(True)
                line2.set_visible(True)
                angle_text.set_visible(True)
                # Show corresponding basis vectors
                self.scatter[i].set_visible(True)
                self.line_ori[i].set_visible(True)
                self.line_interp[i].set_visible(True)

        self.canvas.draw_idle()

    def update(self, frame):
        """
        Unified update function for both radar graph and standard visualization.
        """
        if self.current_view == "radar_graph":
            alpha = frame / (self.data['total_frames'] - 1)
            # print(f"Frame {frame}/{self.data['total_frames']}: alpha={alpha:.3f}")  # Debugging: Frame progress

            for i, line in enumerate(self.radar_lines):
                interpolated_row = (1 - alpha) * self.radar_data_start[i] + alpha *  self.radar_data_end[i]

                # Update the radar chart
                line.set_ydata(interpolated_row)
                self.radar_fills[i].remove()
                self.radar_fills[i] = self.radar_ax.fill(self.radar_theta, interpolated_row, alpha=0.25)[0]
            self.radar_ax.legend(bbox_to_anchor=(1.5, 1), loc='upper left', borderaxespad=0)
            
            
            # return self.radar_lines + self.radar_fills

        elif self.current_view == "standard_view":
            alpha = frame / (self.data['total_frames'] - 1) if self.data['total_frames'] > 1 else 1.0
            # Update standard visualization (basis vectors and angles)
            for i, (line_interp, line_ori, scatter) in enumerate(zip(self.line_interp, self.line_ori, self.scatter)):
                interpolated_data = (1 - alpha) * self.Basis_interp_line[i] + alpha * self.target_interp_line[i]
                interpolated_line_ori = (1 - alpha) * self.Basis[i] + alpha * self.data['target_Basis'][i]

                # Handle non-finite values
                if not np.all(np.isfinite(interpolated_data)):
                    interpolated_data = np.nan_to_num(interpolated_data, nan=0.0, posinf=0.0, neginf=0.0)
                if not np.all(np.isfinite(interpolated_line_ori)):
                    interpolated_line_ori = np.nan_to_num(interpolated_line_ori, nan=0.0, posinf=0.0, neginf=0.0)

                line_interp.set_ydata(interpolated_data)
                line_ori.set_ydata(interpolated_line_ori)
                interpolated_offsets = np.column_stack((self.data['vec_dimension'], interpolated_line_ori))
                scatter.set_offsets(interpolated_offsets)

            # self.ax_top.relim()
            self.ax_top.autoscale_view()
            self.ax_top.grid(True)
            self.ax_top.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            self.ax_top.set_xticks(self.data['vec_dimension'])

            # Update angles (arcs)
            for i, (arc, line1, line2, angle_text, initial_angle, target_angle) in enumerate(self.angle_plots):
                if i >= len(self.angle_plots):
                    continue
                offset = get_arc_offset(i, self.Basis.shape[0])
                update_angle(frame, self.data['total_frames'], self.ax_angles, i, arc, line1, line2, angle_text,
                            initial_angle, target_angle, radius=0.4, offset=offset)

        # Update the slider position only if the animation is running
        if self.anim.event_source is not None:
            self.slider_bar.setValue(frame)

        self.canvas.draw_idle()


    def animation(self):

        ''' Initialize and start the animation '''
        self.anim = FuncAnimation(
            self.fig,
            self.update,
            frames=self.data['total_frames'],
            interval=100,
            repeat=self.repeat,
            blit=False
        )
        
        if self.current_view == "standard_view":
            self.ax_top.grid(True)

        self.canvas.draw()

    def on_toggle(self):
        '''Toggle between play and pause states'''
        if self.play_button.text() == 'Pause':
            self.anim.event_source.stop()
            self.play_button.setText('Play')
        else:
            self.anim.event_source.start()
            self.play_button.setText('Pause')

    def on_slider(self):
        ''' Function to control the slider '''
        new_frame = self.slider_bar.value()
        self.anim.event_source.stop()
        self.update(new_frame)
        self.canvas.draw_idle()

        if self.play_button.text() == 'Pause':
            self.anim.event_source.start()

    def on_checkbox_toggle(self):
        '''Toggle the repeat option of the animation'''
        self.repeat = self.loop_checkbox.isChecked()
        self.anim.repeat = self.repeat

    def closeEvent(self, event):
        ''' Properly close the animation to prevent errors '''
        self.anim.event_source.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    ex.show()
    sys.exit(app.exec_())
