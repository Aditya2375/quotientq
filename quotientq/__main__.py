import argparse
import json
from .core import analyze


def main():
    parser = argparse.ArgumentParser(
        description="Synthetic sampled Q-learning market-making experiment"
    )
    parser.add_argument("--stress", type=int, default=0)
    parser.add_argument("--budget", type=int, default=0)
    args = parser.parse_args()
    try:
        result = analyze(vars(args))
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
