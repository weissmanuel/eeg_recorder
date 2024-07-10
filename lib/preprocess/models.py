from enum import Enum


class ProcessStage(Enum):
    RECORDING = 'RECORDING'
    TRAINING = 'TRAINING'
    INFERENCE = 'INFERENCE'

    def __str__(self):
        return self.value()

    @staticmethod
    def from_str(label: str) -> 'ProcessStage':
        return ProcessStage[label.upper()]
