from core.WakeWordDetector import WakeWordDetector
from core.Transcriber import Transcriber
from core.CommandManager import CommandManager
from commands.CmdDefault import CmdDefault

class VoiceAssistantPipeline:
    def __init__(self):
        self.wwd = WakeWordDetector(response_path="../responses/GLaDOS_Okay.wav", model_path="../saved_models/wake_word_conv2d_aug")
        self.trans = Transcriber(model_path="../saved_models/MASR_Model")

        self.cm = CommandManager()
        def_cmd = CmdDefault("Default Command")
        self.cm.add_cmd("default", def_cmd)