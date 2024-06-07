import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
from typing import Callable
from lib.recorder import InletInfo, RecordingInfo
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib import style
from matplotlib import animation
from lib.store import RealTimeStore, PlotStore
import copy

style.use("ggplot")


class KeyValue:

    def __init__(self, parent, key: str, value: str, row: int = 0, value_anchor='e'):
        self.key = ctk.CTkLabel(parent, text=key, anchor='w')
        self.key.grid(row=row, column=0, pady=5, padx=10, sticky='nsew')
        self.value = ctk.CTkLabel(parent, text=value, anchor=value_anchor)
        self.value.grid(row=row, column=1, pady=5, padx=10, sticky='nsew')

    def update_key(self, text: str):
        self.key.configure(text=text)

    def update_value(self, text: str):
        self.value.configure(text=text)


def safe_update_value(attribute: KeyValue, value: any):
    if value is not None:
        if isinstance(value, float):
            value = round(value, 4)
        attribute.update_value(str(value))


class StreamInfo(ctk.CTkFrame):

    def __init__(self, master, title: str):
        super().__init__(master)

        self.grid(pady=(0, 20))

        self.title = ctk.CTkLabel(self, text=title, anchor='n', font=ctk.CTkFont(weight='bold'))
        self.title.grid(row=0, column=0, pady=(20, 0), padx=5, sticky='nsew')

        self.source_id = KeyValue(self, key='Source ID', value='', row=1)
        self.sfreq = KeyValue(self, key='Sample Frequency:', value='0', row=2)
        self.n_channels = KeyValue(self, key='# Channels:', value='0', row=3)
        self.samples_recorded = KeyValue(self, key='Samples Recorded:', value='0', row=4)
        self.samples_expected = KeyValue(self, key='Samples Expected:', value='0', row=5)
        self.time_shift = KeyValue(self, key='Time Shift:', value='0', row=6)
        self.iterations = KeyValue(self, key='Iterations:', value='0', row=7)

    def update_info(self, info: InletInfo):
        safe_update_value(self.source_id, info.source_id)
        safe_update_value(self.sfreq, info.sfreq)
        safe_update_value(self.n_channels, info.n_channels)
        safe_update_value(self.time_shift, info.time_shift)
        safe_update_value(self.samples_recorded, info.samples_recorded)
        safe_update_value(self.samples_expected, info.samples_expected)
        safe_update_value(self.iterations, info.iterations)


class Header(ctk.CTkFrame):

    def __init__(self, parent):
        super().__init__(parent)

        self.grid(row=0, column=0, padx=10, pady=(10, 0), sticky='nsew')
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.header_box_left = ctk.CTkFrame(self, fg_color='transparent')
        self.header_box_left.grid(row=0, column=0)
        self.header_box_right = ctk.CTkFrame(self, fg_color='transparent')
        self.header_box_right.grid(row=0, column=1)

        self.status_label = ctk.CTkLabel(self.header_box_left, text="Status:", font=ctk.CTkFont(weight='bold'))
        self.status_label.grid(row=0, column=0, pady=10, padx=10, sticky='s')

        self.status_value = ctk.CTkLabel(self.header_box_right, text="Idle", font=ctk.CTkFont(weight='bold'))
        self.status_value.grid(row=0, column=1, pady=10, padx=10, sticky='s')

        self.start_button = ctk.CTkButton(self.header_box_left, text="Start",
                                          command=lambda: print("Starting EEG Recorder"))
        self.start_button.grid(row=1, column=0, pady=10, padx=10, sticky='e')

        self.stop_button = ctk.CTkButton(self.header_box_right, text="Stop",
                                         command=lambda: print("Stopping EEG Recorder"))
        self.stop_button.grid(row=1, column=1, pady=10, padx=10, sticky='e')


class Body(ctk.CTkFrame):

    def __init__(self, parent):
        super().__init__(parent, fg_color='transparent')

        self.grid(row=1, column=0, sticky='nsew', padx=10, pady=(10, 0))
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        self.signal_section = StreamInfo(self, 'Signal Information')
        self.signal_section.grid(row=0, column=0, sticky='nsew', padx=(0, 5), pady=0)

        self.marker_section = StreamInfo(self, 'Marker Information')
        self.marker_section.grid(row=0, column=1, sticky='nsew', padx=(5, 0), pady=0)


class Footer(ctk.CTkFrame):

    def __init__(self, parent, plot_store: PlotStore = None):
        super().__init__(parent)

        self.plot_store = plot_store

        self.grid(row=2, column=0, sticky='nsew', padx=10, pady=20)

        self.title = ctk.CTkLabel(self, text="EEG Recording Info", anchor='n', font=ctk.CTkFont(weight='bold'))
        self.title.grid(row=0, column=0, pady=(20, 0), padx=10, sticky='nsew')

        self.start_time = KeyValue(self, key='Start Time:', value='', row=1)
        self.end_time = KeyValue(self, key='End Time:', value='', row=2)
        self.duration = KeyValue(self, key='Duration:', value='', row=3)
        self.file_path = KeyValue(self, key='File Path:', value='', row=4)

        self.label = ctk.CTkLabel(self, text="Graph View", anchor='n', font=ctk.CTkFont(weight='bold'))
        self.label.grid(row=5, column=0, pady=(20, 0), padx=10, sticky='nsew')
        self.figure = Figure(figsize=(5, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.plot(np.random.rand(10))
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().grid(row=6, column=0, pady=10, padx=10, sticky='nsew')
        self.animation = animation.FuncAnimation(self.figure, func=self.update_graph, interval=50, cache_frame_data=False)

    def update_info(self, info: RecordingInfo):
        self.start_time.update_value(str(info.start_time))
        self.end_time.update_value(str(info.end_time))
        self.duration.update_value(str(info.duration))
        self.file_path.update_value(str(info.file_path))

    # def update_graph(self, i):
    #     try:
    #         with open("./data/real_time/data.npy", "rb") as f:
    #             x = np.load(f)
    #             y = np.load(f)
    #             y = y[0, :]
    #             self.ax.clear()
    #             self.ax.plot(x, y)
    #     except Exception as e:
    #         pass

    def update_graph(self, i):
        try:
            if self.plot_store is not None:
                x, y = copy.deepcopy(self.plot_store.get_data())
                self.ax.clear()
                self.ax.plot(x, y)

        except Exception as e:
            pass

    def stop(self):
        try:
            self.animation.pause()
        except Exception:
            pass


class Interface(ctk.CTk):

    plot_store: PlotStore | None = None

    def __init__(self, geometry: str = '600x700', plot_store: PlotStore = None):
        super().__init__()

        self.plot_store = plot_store

        ctk.set_appearance_mode("dark")
        self.title("EEG Recorder")
        self.geometry(geometry)
        self.iconbitmap("./assets/favicon.ico")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.root = ctk.CTkScrollableFrame(self, width=100000, height=800, fg_color='transparent')
        self.root.grid(column=0, row=0)
        self.root.grid_columnconfigure(0, weight=1)

        self.header = Header(self.root)
        self.body = Body(self.root)
        self.footer = Footer(self.root, self.plot_store)

        # self.graph_view = ctk.CTkToplevel(self)

    def set_start_action(self, start_action: Callable) -> 'Interface':
        self.header.start_button.configure(command=lambda: self.after(10, start_action))
        return self

    def set_stop_action(self, stop_action: Callable) -> 'Interface':
        self.header.stop_button.configure(command=lambda: self.after(10, stop_action))
        return self

    def set_status(self, text: str) -> 'Interface':
        self.header.status_value.configure(text=text)
        return self

    def set_recording_info(self, info: RecordingInfo) -> 'Interface':
        if info is not None:
            if info.signal_info is not None:
                self.body.signal_section.update_info(info.signal_info)
            if info.marker_info is not None:
                self.body.marker_section.update_info(info.marker_info)
        self.footer.update_info(info)
        return self

    def set_signal_progress(self, info: InletInfo) -> None:
        self.body.signal_section.update_info(info)

    def set_marker_progress(self, info: InletInfo) -> None:
        self.body.marker_section.update_info(info)

    def run(self):
        self.mainloop()

    def stop(self):
        try:
            self.footer.stop()
            self.destroy()
        except Exception as e:
            pass
