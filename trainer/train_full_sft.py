import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trainer.train_text import main

if __name__ == "__main__":
    main("sft")
