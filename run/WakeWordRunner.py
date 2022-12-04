from core.WakeWordDetector import WakeWordDetector
import os
import sys


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    wwd = WakeWordDetector(response_path=r"..\responses\GLaDOS_Okay.wav")
    event = wwd.listen()
    while event.wait():
        print("Runner detected Response!")
        wwd.stop()
        sys.exit(0)


if __name__ == "__main__":
    main()
