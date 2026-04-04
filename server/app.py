"""OpenEnv-compatible server entrypoint."""

from app import app as root_app
from app import main as root_main

app = root_app


def main(host: str = "0.0.0.0", port: int = 7860):
    root_main(host=host, port=port)


if __name__ == "__main__":
    main()
