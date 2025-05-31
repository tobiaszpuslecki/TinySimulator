class Model:
    def __init__(self, model_info):
        self.id = model_info.get("id")
        self.training_accuracy = model_info.get("training_accuracy")
        # self.energy_consumption = model_info.get('energy_consumption') # moved to config
        # self.ram_consumption = model_info.get('ram_consumption')
        # self.flash_consumption = model_info.get('flash_consumption')
        self.latency = model_info.get("latency")
        # self.macc = model_info.get('macc')
        self.internal_counter = 0

    def infer(self, probe=None):
        self.internal_counter += 1
        pred = 66
        return pred

    def get_inference_no(self):
        return self.internal_counter

    def clear_inference_no(self):
        self.internal_counter = 0
