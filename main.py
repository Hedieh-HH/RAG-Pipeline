import logging
import sys

from myrag.config import get_settings
from myrag.pipeline import Pipeline


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


def main() -> None:
    settings = get_settings()
    _setup_logging(settings.logging.level)

    result = Pipeline(settings).run()

    print(f"\nPipeline complete")
    print(f"  Total   : {result.total}")
    print(f"  Indexed : {result.indexed}")
    print(f"  Skipped : {result.skipped}")
    print(f"  Failed  : {result.failed}")

    if result.failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
