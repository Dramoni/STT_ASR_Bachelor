from commands.CmdDefault import CmdDefault


class CommandManager:
    def __init__(self):
        self.cmds = {}

    def add_cmd(self, key, cmd):
        self.cmds[key] = cmd

    def run_cmd(self, transcript):
        for key in self.cmds:
            if key in transcript:
                self.cmds[key].run(transcript)
                return

        print(f"[INFO] Command not found for: {transcript}")

'''
### EXAMPLE USAGE ###

cm = CommandManager()

default = CmdDefault("Default Command")
default.summary()
cm.add_cmd("default", default)
cm.run_cmd("führe bitte das default kommando aus")
'''
