from core.WakeWordDetector import WakeWordDetector
import os


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    wwd = WakeWordDetector(response_path=r"E:\Bachelor\GLaDOS_Okay.wav")
    event = wwd.listen()
    while event.wait():
        print("Runner detected Response!")


if __name__ == "__main__":
    main()
