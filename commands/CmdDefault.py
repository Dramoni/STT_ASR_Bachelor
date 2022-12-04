from core.ACommand import ACommand


class CmdDefault(ACommand):
    def __init__(self, name: str):
        super().__init__(name, "This is the default command.")

    def run(self):
        super().run()
        print("Hello World!")
