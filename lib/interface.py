import customtkinter as tk
from typing import Callable
from lib.recorder import InletInfo, RecordingInfo
import numpy as np


class KeyValue:

    def __init__(self, parent, key: str, value: str, row: int = 0, value_anchor='e'):
        self.key = tk.CTkLabel(parent, text=key, anchor='w')
        self.key.grid(row=row, column=0, pady=5, padx=10, sticky='nsew')
        self.value = tk.CTkLabel(parent, text=value, anchor=value_anchor)
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


class StreamInfo(tk.CTkFrame):

    def __init__(self, master, title: str):
        super().__init__(master)

        self.grid(pady=(0, 20))

        self.title = tk.CTkLabel(self, text=title, anchor='n', font=tk.CTkFont(weight='bold'))
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


class Header(tk.CTkFrame):

    def __init__(self, parent):
        super().__init__(parent)

        self.grid(row=0, column=0, padx=10, pady=(10, 0), sticky='nsew')
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.header_box_left = tk.CTkFrame(self, fg_color='transparent')
        self.header_box_left.grid(row=0, column=0)
        self.header_box_right = tk.CTkFrame(self, fg_color='transparent')
        self.header_box_right.grid(row=0, column=1)

        self.status_label = tk.CTkLabel(self.header_box_left, text="Status:", font=tk.CTkFont(weight='bold'))
        self.status_label.grid(row=0, column=0, pady=10, padx=10, sticky='s')

        self.status_value = tk.CTkLabel(self.header_box_right, text="Idle", font=tk.CTkFont(weight='bold'))
        self.status_value.grid(row=0, column=1, pady=10, padx=10, sticky='s')

        self.start_button = tk.CTkButton(self.header_box_left, text="Start",
                                         command=lambda: print("Starting EEG Recorder"))
        self.start_button.grid(row=1, column=0, pady=10, padx=10, sticky='e')

        self.stop_button = tk.CTkButton(self.header_box_right, text="Stop",
                                        command=lambda: print("Stopping EEG Recorder"))
        self.stop_button.grid(row=1, column=1, pady=10, padx=10, sticky='e')


class Body(tk.CTkFrame):

    def __init__(self, parent):
        super().__init__(parent, fg_color='transparent')

        self.grid(row=1, column=0, sticky='nsew', padx=10, pady=(10, 0))
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        self.signal_section = StreamInfo(self, 'Signal Information')
        self.signal_section.grid(row=0, column=0, sticky='nsew', padx=(0, 5), pady=0)

        self.marker_section = StreamInfo(self, 'Marker Information')
        self.marker_section.grid(row=0, column=1, sticky='nsew', padx=(5, 0), pady=0)


class Footer(tk.CTkFrame):

    def __init__(self, parent):
        super().__init__(parent)

        self.grid(row=2, column=0, sticky='nsew', padx=10, pady=20)

        self.title = tk.CTkLabel(self, text="EEG Recording Info", anchor='n', font=tk.CTkFont(weight='bold'))
        self.title.grid(row=0, column=0, pady=(20, 0), padx=10, sticky='nsew')

        self.start_time = KeyValue(self, key='Start Time:', value='', row=1)
        self.end_time = KeyValue(self, key='End Time:', value='', row=2)
        self.duration = KeyValue(self, key='Duration:', value='', row=3)
        self.file_path = KeyValue(self, key='File Path:', value='', row=4)

    def update_info(self, info: RecordingInfo):
        self.start_time.update_value(str(info.start_time))
        self.end_time.update_value(str(info.end_time))
        self.duration.update_value(str(info.duration))
        self.file_path.update_value(str(info.file_path))


class Interface(tk.CTk):

    def __init__(self, geometry: str = '600x700'):
        super().__init__()
        tk.set_appearance_mode("dark")
        self.title("EEG Recorder")
        self.geometry(geometry)
        self.iconbitmap("./assets/favicon.ico")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.root = tk.CTkScrollableFrame(self, width=100000, height=800, fg_color='transparent')
        self.root.grid(column=0, row=0)
        self.root.grid_columnconfigure(0, weight=1)

        self.header = Header(self.root)
        self.body = Body(self.root)
        self.footer = Footer(self.root)

    def set_start_action(self, start_action: Callable) -> 'Interface':
        self.header.start_button.configure(command=start_action)
        return self

    def set_stop_action(self, stop_action: Callable) -> 'Interface':
        self.header.stop_button.configure(command=stop_action)
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
