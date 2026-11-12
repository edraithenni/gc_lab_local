import ray

from config import Config
from parameter_server import ParameterServer


def main():
    config = Config.from_env()
    ray.init(
        address="local",
        num_cpus=config.num_workers + 1,
        resources={
            "gc_parameter_server": 1,
            "gc_worker": config.num_workers,
        },
        include_dashboard=True,
        dashboard_host="127.0.0.1",
        dashboard_port=8265,
    )
    try:
        ray.get(ParameterServer.remote().run_training.remote())
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
