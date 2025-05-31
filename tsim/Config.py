from . import Time


class Config:
    def __init__(
        self,
        use_tqdm_=None,
        mode_=None,
        min_interval_=None,
        reference_=None,
        print_result_=None,
        board_=None,
        battery_capacity_=1,
        timebase=Time.TimeBase.TIMEBASE_0_01_MS.value,
    ):
        self.use_tqdm = use_tqdm_
        self.mode = mode_
        self.min_interval = min_interval_
        self.reference = reference_
        self.print_result = print_result_
        self.board = board_
        # self.battery_capacity = 100 # mAh
        # self.battery_capacity = 1  # mAh
        self.battery_capacity = battery_capacity_

        self.timebase = timebase
        self.random_seed = 1
        self.randomness_tres = 0.5
        self.buffer_size = 6


class Device:
    def __init__(self, device_info):
        # self.id = device_info.get('id')
        # self.ram = device_info.get('ram')
        # self.flash = device_info.get('flash')
        # energy cons. is current in ampers!
        self.avg_idle_energy_cons = device_info.get("avg_idle_energy_cons")
        self.avg_infer_energy_cons = device_info.get("avg_infer_energy_cons")
        # self.busy = False


class EnergySource:
    def __init__(self, prod=None):
        self.active = False
        self.prod = prod

    # if solar.active:
    #     batt.charge(time=1*timebase, current=solar.production)


class Battery:
    # src: https://www.omnicalculator.com/other/battery-life
    def __init__(self, battery_info):
        self.id = battery_info.get("id")
        self.discharge_safety = 0
        # mAh
        self.max_capacity = battery_info.get("max_capacity")
        if battery_info.get("current_capacity") is None:
            self.current_capacity = self.max_capacity
        else:
            self.current_capacity = battery_info.get("current_capacity")

    def percentage(self):
        return self.current_capacity / self.max_capacity

    def discharge(self, time, current):
        capacity = time / 3600 / 1000 * current * (1 - self.discharge_safety)
        self.current_capacity -= capacity

    def charge(self, time, current):
        capacity = time / 3600 / 1000 * current * (1 - self.discharge_safety)
        self.current_capacity += capacity


class Buffer:
    def __init__(self, size):
        self.size = size
        self.elem_no = 0

    def enqueue(self):
        if self.elem_no + 1 <= self.size:
            self.elem_no += 1

    def dequeue(self):
        if self.elem_no != 0:
            self.elem_no -= 1
            return True

        return False

    def get(self):
        return self.elem_no
