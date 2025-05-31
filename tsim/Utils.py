import json
from . import Config
from . import Model


def read_json(filepath):
    with open(filepath + ".json", encoding="UTF-8") as f:
        data = json.load(f)
        models_ = []
        for m in data["models"]:
            models_.append(
                Model.Model(
                    {
                        "id": m["id"],
                        "training_accuracy": m["training_accuracy"],
                        "latency": m["latency"] + data["selector_latency"],
                    }
                )
            )
    return models_


def configure_board(filepath):
    with open(filepath + ".json", encoding="UTF-8") as f:
        data = json.load(f)
        # print(data)
        board = Config.Device(
            {
                "avg_idle_energy_cons": data["avg_idle_energy_cons"],
                "avg_infer_energy_cons": data["avg_infer_energy_cons"],
            }
        )
        return board
