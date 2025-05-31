import enum
import numpy as np
import random


class Mode(enum.Enum):
    MODE_SYNC = "sync"
    MODE_ASYNC = "async"


class TimeBase(float, enum.Enum):
    TIMEBASE_MS = 1  # 1 ms
    TIMEBASE_0_1_MS = 0.1  # 0.1 ms
    TIMEBASE_0_01_MS = 0.01  # 0.01 ms


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class TimeSource(metaclass=SingletonMeta):
    def __init__(self):
        self._timestamp = 0

    def forward(self, step=1):
        self._timestamp += step


# class RandomGenerator(metaclass=SingletonMeta):
#     def __init__(self, seed, test_ds_size=10):
#         self.test_ds_size = test_ds_size
#         self.seed = seed
#         np.random.seed(seed)
#         random.seed(seed)

#     def next(self, config_):
#         is_event_performed = random.uniform(0.0, 1.0) > config_.randomness_tres
#         if (
#             is_event_performed or config_.mode == Mode.MODE_SYNC
#         ):  # for sync always return random value
#             return np.random.randint(0, self.test_ds_size, 1)[0]

#         return None


# class DataStream(TimeSource):
#     def __init__(self, min_interval, generator):
#         super().__init__()
#         self.min_interval = min_interval
#         self.last_event_timestamp = 0
#         self.generator = generator

#     # TimeSeries.forward() returns if probe is present and its index
#     def forward(self, config_):

#         current_timestamp = self._timestamp

#         if (
#             current_timestamp - self.last_event_timestamp > self.min_interval
#         ):  # or current_timestamp == 0:
#             # generate event
#             rand_value = self.generator.next(config_)
#             if rand_value is not None:
#                 self.last_event_timestamp = current_timestamp
#             res = (current_timestamp, rand_value)
#         else:
#             res = (current_timestamp, None)

#         # TimeSource forward
#         super().forward()
#         return res


class RandomGenerator(metaclass=SingletonMeta):
    def __init__(self, seed, test_ds_size=10):
        self.test_ds_size = test_ds_size
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def next(self, config_):
        is_event_performed = random.uniform(0.0, 1.0) > config_.randomness_tres
        if (
            is_event_performed or config_.mode == Mode.MODE_SYNC
        ):  # for sync always return random value
            return True
        return None


class DataStream(TimeSource):
    def __init__(self, min_interval, generator):
        super().__init__()
        self.min_interval = min_interval
        self.last_event_timestamp = 0
        self.generator = generator

    def forward(self, config_):

        current_timestamp = self._timestamp

        if current_timestamp - self.last_event_timestamp > self.min_interval:
            # generate event
            rand_value = self.generator.next(config_)
            if rand_value is not None:
                self.last_event_timestamp = current_timestamp
            res = rand_value
        else:
            res = None

        # TimeSource forward
        super().forward()
        return res
