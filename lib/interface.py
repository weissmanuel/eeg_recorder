import customtkinter as tk
from typing import Callable
from lib.recorder import InletInfo
import numpy as np


class KeyValue:

    def __init__(self, parent, key: str, value: str, row: int = 0):
        self.key = tk.CTkLabel(parent, text=key, anchor='w')
        self.key.grid(row=row, column=0, pady=5, padx=10, sticky='nsew')
        self.value = tk.CTkLabel(parent, text=value, anchor='e')
        self.value.grid(row=row, column=1, pady=5, padx=10, sticky='nsew')

    def update_key(self, text: str):
        self.key.configure(text=text)

    def update_value(self, text: str):
        self.value.configure(text=text)


class StreamInfo(tk.CTkFrame):

    def __init__(self, master, title: str):
        super().__init__(master)

        self.title = KeyValue(self, key=title, value='', row=0)
        self.total = KeyValue(self, key='Total', value='0', row=1)
        self.received = KeyValue(self, key='Last Received:', value='0', row=2)
        self.time_shift = KeyValue(self, key='Time Shift:', value='0', row=3)
        self.iteration = KeyValue(self, key='Iteration:', value='0', row=4)

    def update_info(self, info: InletInfo):
        self.total.update_value(str(info.n_total))
        self.received.update_value(str(info.n_received))
        self.time_shift.update_value(str(np.round(info.time_shift, 4)))
        self.iteration.update_value(str(info.iteration))


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

        self.status_label = tk.CTkLabel(self.header_box_left, text="Status:")
        self.status_label.grid(row=0, column=0, pady=10, padx=10, sticky='s')

        self.status_value = tk.CTkLabel(self.header_box_right, text="Idle")
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

        self.grid(row=2, column=0, sticky='nsew', padx=10, pady=10)


class Interface(tk.CTk):

    def __init__(self, geometry: str = '600x600'):
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

    def set_signal_progress(self, info: InletInfo) -> None:
        self.body.signal_section.update_info(info)


    def set_marker_progress(self, info: InletInfo) -> None:
        self.body.marker_section.update_info(info)

    def run(self):
        self.mainloop()
