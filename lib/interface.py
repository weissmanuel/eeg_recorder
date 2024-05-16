import tkinter as tk
from typing import Callable


class Interface:

    def __init__(self, geometry: str = '600x200'):
        self.root = tk.Tk()
        self.root.geometry(geometry)
        self.root.title("EEG Recorder")
        self.root.iconbitmap("./assets/favicon.ico")

        self.status_label = tk.Label(self.root, text="Status:")
        self.status_label.grid(row=0, column=0, pady=10, padx=10, sticky='s')

        self.status_value = tk.Label(self.root, text="Idle")
        self.status_value.grid(row=0, column=1, pady=10, padx=10, sticky='s')

        self.start_button = tk.Button(self.root, text="Start", command=lambda: print("Starting EEG Recorder"), width=20,
                                      height=5)
        self.start_button.grid(row=1, column=0, pady=10, padx=10, sticky='e')

        self.stop_button = tk.Button(self.root, text="Stop", command=lambda: print("Stopping EEG Recorder"), width=20,
                                     height=5)
        self.stop_button.grid(row=1, column=1, pady=10, padx=10, sticky='e')

    def set_start_action(self, start_action: Callable) -> 'Interface':
        self.start_button.config(command=start_action)
        return self

    def set_stop_action(self, stop_action: Callable) -> 'Interface':
        self.stop_button.config(command=stop_action)
        return self

    def set_status(self, text: str) -> 'Interface':
        self.status_value.config(text=text)
        return self

    def run(self):
        self.root.mainloop()
