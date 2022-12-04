from abc import ABC, abstractmethod


class ACommand(ABC):
    def __init__(self, name: str, summary: str = "There is no summary :("):
        self.name = name
        self.summary_txt = summary

    def summary(self):
        header = f"{'-' * 20} Summary {self.name} {'-' * 20}"
        print(header)
        print(self.summary_txt)
        print("-" * len(header))

    def run(self):
        print(f"[INFO] Running {self.name}...")
