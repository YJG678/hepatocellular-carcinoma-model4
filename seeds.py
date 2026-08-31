"""Every random seed used by the reproducibility archive."""

from config import SEED_2D

ALL = {"SEED_2D": SEED_2D}


if __name__ == "__main__":
    for name, value in ALL.items():
        print(f"{name} = {value}")

