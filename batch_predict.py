import argparse
import logging
import random
import time

from apscheduler.schedulers.blocking import BlockingScheduler
from sklearn.datasets import load_iris

from db import (
    fetch_unpredicted,
    init_db,
    insert_input,
    insert_prediction,
)
from predict import predict

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("batch")


def run_batch():
    init_db()
    rows = fetch_unpredicted()
    if not rows:
        log.info("no unpredicted rows")
        return
    log.info("processing %d rows", len(rows))
    for row in rows:
        features = [
            row["sepal_length"],
            row["sepal_width"],
            row["petal_length"],
            row["petal_width"],
        ]
        result = predict(features)
        insert_prediction(
            row["id"], result["predicted_class"], result["predicted_label"]
        )
        log.info("id=%s -> %s", row["id"], result["predicted_label"])


def seed(n: int):
    init_db()
    iris = load_iris()
    rng = random.Random(int(time.time()))
    for _ in range(n):
        sample = iris.data[rng.randrange(len(iris.data))]
        insert_input([float(x) for x in sample])
    log.info("seeded %d rows", n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=["run", "schedule", "seed", "init"], help="action"
    )
    parser.add_argument("--n", type=int, default=10, help="rows to seed")
    parser.add_argument(
        "--interval", type=int, default=300, help="schedule interval (seconds)"
    )
    args = parser.parse_args()

    if args.command == "init":
        init_db()
        log.info("db initialized")
    elif args.command == "seed":
        seed(args.n)
    elif args.command == "run":
        run_batch()
    elif args.command == "schedule":
        init_db()
        scheduler = BlockingScheduler()
        scheduler.add_job(
            run_batch, "interval", seconds=args.interval, next_run_time=None
        )
        run_batch()
        log.info("scheduler started, interval=%ss", args.interval)
        scheduler.start()


if __name__ == "__main__":
    main()
