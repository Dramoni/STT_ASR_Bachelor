import pyautogui as gui
import time
import os
import subprocess


class InputManager:
    def __init__(self):
        self.command_dict = {
            -2: self.__unkown_command,
            -1: self.__test,
            0: self.__open,
            1: self.__mvdir_back,
            2: self.__cpy_file,
            3: self.__paste_file,
            4: self.__write,
            5: self.__run_proc,
            6: self.__switch_window,
            7: self.__close_window,
            8: self.__select_next,
        }

    def execute_command(self, cmd: int = -2, args: list = None):
        print(f"Running command {cmd}")
        self.command_dict[cmd](args)

    def __unkown_command(self, args: list):
        print("Got unknown command :(")

    def __test(self, args):
        print("Running test (Alt + Tab)")
        gui.keyDown('alt')
        gui.press('tab')
        time.sleep(1)
        gui.keyUp('alt')

    def __open(self, args: list):
        print("Running open directory")

        try:
            subprocess.Popen(r'explorer /select, ')
        except Exception as e:
            print(f"An error occoured: {e}")

    def __mvdir_back(self, args: list):
        print("Running move back")

        gui.press('backspace')

    def __cpy_file(self, args: list):
        print("Running copy file")

    def __paste_file(self, args: list):
        print("Running paste file")

    def __write(self, args: list):
        print("Running write")
        if args is None or type(args[0]) is not str:
            print(f"Got wrong arguments ({type(args[0]) if args is not None else 'None'})")
            return

        gui.typewrite(args[0], interval=0.1)

    def __run_proc(self, args: list):
        print("Running run process")

    def __switch_window(self, args: list):
        print("Running switch window")

        gui.keyDown('alt')
        gui.press('tab')
        time.sleep(2)
        gui.keyUp('alt')

    def __close_window(self, args: list):
        print("Running close window")

    def __select_next(self, args: list):
        print("Running select next (Tabulator)")

        gui.press('tab')


im = InputManager()
im.execute_command(cmd=7)
im.execute_command(cmd=9)
