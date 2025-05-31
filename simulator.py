import random
import math
import argparse
import unittest
import enum
import json
import numpy as np
import simpful as sp
from tqdm import tqdm
import tsim


class Selector:
    def __init__(self, selector, models_set=None, buffer_size=None):
        self.buffer_size = buffer_size
        self.selector = selector
        self.models_set = models_set

    def select(self, battery, queue):
        return self.selector.select(battery, queue)


class ReferenceSelectors:
    def __init__(self, models_set, strategy):
        self.models_set = models_set
        self.strategy = strategy

    def select(self, battery=None, queue=None):
        if self.strategy == "random":
            return np.random.randint(0, len(self.models_set))
        elif self.strategy == "latency-first":
            return self.models_set.index(
                min(self.models_set, key=lambda model: model.latency)
            )
        elif self.strategy == "accuracy-first":
            return self.models_set.index(
                max(self.models_set, key=lambda model: model.training_accuracy)
            )


class FuzzySystemHolderTova:
    def __init__(self):
        self.fuzzy_system = sp.FuzzySystem(show_banner=False)

        battety_sets = []
        battety_sets.append(
            sp.FuzzySet(function=sp.Triangular_MF(a=0, b=0, c=25), term="low")
        )
        battety_sets.append(
            sp.FuzzySet(function=sp.Triangular_MF(a=0, b=25, c=50), term="medium")
        )
        battety_sets.append(
            sp.FuzzySet(function=sp.Triangular_MF(a=25, b=50, c=75), term="high")
        )
        battety_sets.append(
            sp.FuzzySet(
                function=sp.Trapezoidal_MF(a=50, b=75, c=100, d=100), term="full"
            )
        )
        battery_lv = sp.LinguisticVariable(
            battety_sets, concept="Battery percentage", universe_of_discourse=[0, 100]
        )
        # battery_lv.plot()
        self.fuzzy_system.add_linguistic_variable("Battery", battery_lv)

        queue_sets = []
        queue_sets.append(
            sp.FuzzySet(function=sp.Triangular_MF(a=0, b=0, c=2), term="empty")
        )
        queue_sets.append(
            sp.FuzzySet(function=sp.Triangular_MF(a=1, b=2, c=5), term="half")
        )
        queue_sets.append(
            sp.FuzzySet(function=sp.Trapezoidal_MF(a=3, b=5, c=6, d=6), term="full")
        )
        queue_lv = sp.LinguisticVariable(
            queue_sets, concept="Queue state", universe_of_discourse=[0, 6]
        )
        # queue_lv.plot()
        self.fuzzy_system.add_linguistic_variable("Queue", queue_lv)

        self.fuzzy_system.set_crisp_output_value("small", 3)
        self.fuzzy_system.set_crisp_output_value("average", 5)
        self.fuzzy_system.set_crisp_output_value("high", 7)
        self.fuzzy_system.set_crisp_output_value("C", 10)

        rules_set = []
        rules_set.append("IF (Battery IS low) THEN (Treshold IS small)")
        rules_set.append("IF (Battery IS medium) THEN (Treshold IS average)")
        rules_set.append("IF (Battery IS high) THEN (Treshold IS high)")
        rules_set.append("IF (Battery IS full) THEN (Treshold IS C)")

        rules_set.append(
            "IF (Queue IS empty) AND (Battery IS full) THEN (Treshold IS C)"
        )
        rules_set.append(
            "IF (Queue IS half) AND (Battery IS full) THEN (Treshold IS high)"
        )
        rules_set.append(
            "IF (Queue IS full) AND (Battery IS full) THEN (Treshold IS average)"
        )

        rules_set.append(
            "IF (Queue IS empty) AND (Battery IS high) THEN (Treshold IS high)"
        )
        rules_set.append(
            "IF (Queue IS half) AND (Battery IS high) THEN (Treshold IS average)"
        )
        rules_set.append(
            "IF (Queue IS full) AND (Battery IS high) THEN (Treshold IS small)"
        )

        rules_set.append(
            "IF (Queue IS empty) AND (Battery IS medium) THEN (Treshold IS average)"
        )
        rules_set.append(
            "IF (Queue IS half) AND (Battery IS medium) THEN (Treshold IS small)"
        )
        rules_set.append(
            "IF (Queue IS full) AND (Battery IS medium) THEN (Treshold IS small)"
        )

        rules_set.append(
            "IF (Queue IS empty) AND (Battery IS low) THEN (Treshold IS small)"
        )
        rules_set.append(
            "IF (Queue IS half) AND (Battery IS low) THEN (Treshold IS small)"
        )
        rules_set.append(
            "IF (Queue IS full) AND (Battery IS low) THEN (Treshold IS small)"
        )

        self.fuzzy_system.add_rules(rules_set)

    def infer(self, b, q):
        self.fuzzy_system.set_variable("Battery", b)
        self.fuzzy_system.set_variable("Queue", q)
        return self.fuzzy_system.Sugeno_inference(["Treshold"]).get("Treshold")

    def select(self, battery, queue):
        decision = self.infer(battery, queue)
        decision = round(decision)
        decision = 0 if decision == 10 else decision  # classic mode
        return int(decision)

    def print_test(self):

        for x in range(21):
            b, q = 5 * x, 0
            print(f"---B: {b}, Q: {q} - {self.infer(b,q)/10:.2f}")

        print("---^--" * 10)

        b, q = 0, 1
        print(f"---B: {b}, Q: {q} - {self.infer(b,q)/10:.2f}")
        b, q = 0, 3
        print(f"---B: {b}, Q: {q} - {self.infer(b,q)/10:.2f}")
        b, q = 0, 5
        print(f"---B: {b}, Q: {q} - {self.infer(b,q)/10:.2f}")

        print("-----" * 10)

        b, q = 20, 0
        print(f"---B: {b}, Q: {q} - {self.infer(b,q)/10:.2f}")
        b, q = 20, 3
        print(f"---B: {b}, Q: {q} - {self.infer(b,q)/10:.2f}")
        b, q = 20, 5
        print(f"---B: {b}, Q: {q} - {self.infer(b,q)/10:.2f}")

        print("-----" * 10)

        b, q = 30, 0
        print(f"---B: {b}, Q: {q} - {self.infer(b,q)/10:.2f}")
        b, q = 30, 3
        print(f"---B: {b}, Q: {q} - {self.infer(b,q)/10:.2f}")
        b, q = 30, 5
        print(f"---B: {b}, Q: {q} - {self.infer(b,q)/10:.2f}")

        print("-----" * 10)

        b, q = 80, 0
        print(f"---B: {b}, Q: {q} - {self.infer(b,q)/10:.2f}")
        b, q = 80, 3
        print(f"---B: {b}, Q: {q} - {self.infer(b,q)/10:.2f}")
        b, q = 80, 5
        print(f"---B: {b}, Q: {q} - {self.infer(b,q)/10:.2f}")


class FuzzySystemHolderStream:
    def __init__(self):
        self.fuzzy_system = sp.FuzzySystem(show_banner=False)

        battety_sets = []
        battety_sets.append(
            sp.FuzzySet(function=sp.Triangular_MF(a=0, b=0, c=25), term="low")
        )
        battety_sets.append(
            sp.FuzzySet(function=sp.Triangular_MF(a=0, b=25, c=50), term="medium")
        )
        battety_sets.append(
            sp.FuzzySet(function=sp.Triangular_MF(a=25, b=50, c=75), term="high")
        )
        battety_sets.append(
            sp.FuzzySet(
                function=sp.Trapezoidal_MF(a=50, b=75, c=100, d=100), term="full"
            )
        )
        battery_lv = sp.LinguisticVariable(
            battety_sets, concept="Battery percentage", universe_of_discourse=[0, 100]
        )
        # battery_lv.plot()
        self.fuzzy_system.add_linguistic_variable("Battery", battery_lv)

        queue_sets = []
        queue_sets.append(
            sp.FuzzySet(function=sp.Triangular_MF(a=0, b=0, c=2), term="empty")
        )
        queue_sets.append(
            sp.FuzzySet(function=sp.Triangular_MF(a=1, b=2, c=5), term="half")
        )
        queue_sets.append(
            sp.FuzzySet(function=sp.Trapezoidal_MF(a=3, b=5, c=6, d=6), term="full")
        )
        queue_lv = sp.LinguisticVariable(
            queue_sets, concept="Queue state", universe_of_discourse=[0, 6]
        )
        # queue_lv.plot()
        self.fuzzy_system.add_linguistic_variable("Queue", queue_lv)

        self.fuzzy_system.set_crisp_output_value("small", 3)
        self.fuzzy_system.set_crisp_output_value("average", 5)
        self.fuzzy_system.set_crisp_output_value("high", 7)
        self.fuzzy_system.set_crisp_output_value("C", 10)

        rules_set = []
        rules_set.append("IF (Battery IS low) THEN (Treshold IS small)")
        rules_set.append("IF (Battery IS medium) THEN (Treshold IS average)")
        rules_set.append("IF (Battery IS high) THEN (Treshold IS high)")
        rules_set.append("IF (Battery IS full) THEN (Treshold IS C)")

        rules_set.append(
            "IF (Queue IS empty) AND (Battery IS full) THEN (Treshold IS C)"
        )
        rules_set.append(
            "IF (Queue IS half) AND (Battery IS full) THEN (Treshold IS high)"
        )
        rules_set.append(
            "IF (Queue IS full) AND (Battery IS full) THEN (Treshold IS average)"
        )

        rules_set.append(
            "IF (Queue IS empty) AND (Battery IS high) THEN (Treshold IS high)"
        )
        rules_set.append(
            "IF (Queue IS half) AND (Battery IS high) THEN (Treshold IS average)"
        )
        rules_set.append(
            "IF (Queue IS full) AND (Battery IS high) THEN (Treshold IS small)"
        )

        rules_set.append(
            "IF (Queue IS empty) AND (Battery IS medium) THEN (Treshold IS average)"
        )
        rules_set.append(
            "IF (Queue IS half) AND (Battery IS medium) THEN (Treshold IS small)"
        )
        rules_set.append(
            "IF (Queue IS full) AND (Battery IS medium) THEN (Treshold IS small)"
        )

        rules_set.append(
            "IF (Queue IS empty) AND (Battery IS low) THEN (Treshold IS small)"
        )
        rules_set.append(
            "IF (Queue IS half) AND (Battery IS low) THEN (Treshold IS small)"
        )
        rules_set.append(
            "IF (Queue IS full) AND (Battery IS low) THEN (Treshold IS small)"
        )

        self.fuzzy_system.add_rules(rules_set)

    def infer(self, b, q):
        self.fuzzy_system.set_variable("Battery", b)
        self.fuzzy_system.set_variable("Queue", q)
        return self.fuzzy_system.Sugeno_inference(["Treshold"]).get("Treshold")

    def select(self, battery, queue):
        decision = self.infer(battery, queue)
        # if decision <= 3:
        #     return 0
        # elif decision <=5:
        #     return 1
        # elif decision <=7:
        #     return 2
        # return 3

        if decision <= 3:
            return 0
        elif decision <= 5:
            return 1
        return 2

    def print_test(self):

        for x in range(21):
            b, q = 5 * x, 0
            print(f"---B: {b}, Q: {q} - {self.select(b,q)}")

        print("---^--" * 10)

        b, q = 0, 1
        print(f"---B: {b}, Q: {q} - {self.select(b,q)}")
        b, q = 0, 3
        print(f"---B: {b}, Q: {q} - {self.select(b,q)}")
        b, q = 0, 5
        print(f"---B: {b}, Q: {q} - {self.select(b,q)}")

        print("-----" * 10)

        b, q = 20, 0
        print(f"---B: {b}, Q: {q} - {self.select(b,q)}")
        b, q = 20, 3
        print(f"---B: {b}, Q: {q} - {self.select(b,q)}")
        b, q = 20, 5
        print(f"---B: {b}, Q: {q} - {self.select(b,q)}")

        print("-----" * 10)

        b, q = 30, 0
        print(f"---B: {b}, Q: {q} - {self.select(b,q)}")
        b, q = 30, 3
        print(f"---B: {b}, Q: {q} - {self.select(b,q)}")
        b, q = 30, 5
        print(f"---B: {b}, Q: {q} - {self.select(b,q)}")

        print("-----" * 10)

        b, q = 80, 0
        print(f"---B: {b}, Q: {q} - {self.select(b,q)}")
        b, q = 80, 3
        print(f"---B: {b}, Q: {q} - {self.select(b,q)}")
        b, q = 80, 5
        print(f"---B: {b}, Q: {q} - {self.select(b,q)}")


class Simulator:
    def simulate(self, selectorstrategy_, models_, config_):

        max_latency = max(m.latency for m in models_)

        assert not (
            config_.mode == tsim.Time.Mode.MODE_SYNC
            and max_latency > config_.min_interval
        ), "In sync mode min_interval should be larger than max latency!"

        batt = tsim.Config.Battery({"max_capacity": config_.battery_capacity})
        ts = tsim.Time.DataStream(
            min_interval=config_.min_interval / config_.timebase,
            generator=tsim.Time.RandomGenerator(config_.random_seed),
        )
        selector = Selector(selectorstrategy_, buffer_size=config_.buffer_size)
        buffer = tsim.Config.Buffer(config_.buffer_size)

        if config_.use_tqdm:
            pbar = tqdm(total=1.0)

        is_during_inference_semaphore = 0
        iterations = 0
        dropped = 0
        current_model = None
        battery_percentage = batt.percentage()

        if config_.mode == tsim.Time.Mode.MODE_SYNC:
            while battery_percentage > 0:

                battery_percentage = batt.percentage()

                if ts.forward(config_) is not None:  # event happened
                    selection = (
                        selector.select(battery_percentage * 100, 0)
                        if not config_.reference
                        else 0
                    )
                    current_model = models_[selection]
                    current_model.infer()
                    cycles = math.ceil((current_model.latency / config_.timebase))
                    batt.discharge(
                        time=cycles * config_.timebase,
                        current=config_.board.avg_infer_energy_cons,
                    )
                    batt.discharge(
                        time=config_.min_interval * config_.timebase,
                        current=config_.board.avg_idle_energy_cons,
                    )
                else:
                    batt.discharge(
                        time=1 * config_.timebase,
                        current=config_.board.avg_idle_energy_cons,
                    )

                end_batt = batt.percentage()
                if config_.use_tqdm:
                    pbar.update(battery_percentage - end_batt)
                iterations += 1

        if config_.mode == tsim.Time.Mode.MODE_ASYNC:
            while battery_percentage > 0:

                battery_percentage = batt.percentage()

                event_i = ts.forward(config_)

                if is_during_inference_semaphore == 0:
                    if event_i is None:
                        if buffer.get() != 0:
                            buffer.dequeue()
                            # print(f"decueue - {buffer.get()}")
                            selection = (
                                selector.select(battery_percentage * 100, buffer.get())
                                if not config_.reference
                                else 0
                            )
                            current_model = models_[selection]
                            current_model.infer()
                            batt.discharge(
                                time=1 * config_.timebase,
                                current=config_.board.avg_infer_energy_cons,
                            )  # uwzględnić obecny cykl
                            is_during_inference_semaphore = (
                                round((current_model.latency / config_.timebase)) - 1
                            )  # odjąć jeden - zeby uwzględnić obecny cykl
                        else:
                            # print(f"idle")
                            batt.discharge(
                                time=1 * config_.timebase,
                                current=config_.board.avg_idle_energy_cons,
                            )
                    else:
                        # print(f"processing 1")
                        selection = (
                            selector.select(battery_percentage * 100, buffer.get())
                            if not config_.reference
                            else 0
                        )
                        current_model = models_[selection]
                        current_model.infer()
                        batt.discharge(
                            time=1 * config_.timebase,
                            current=config_.board.avg_infer_energy_cons,
                        )  # uwzględnić obecny cykl
                        is_during_inference_semaphore = (
                            round((current_model.latency / config_.timebase)) - 1
                        )  # odjąć jeden - zeby uwzględnić obecny cykl
                else:
                    if event_i is not None:
                        if buffer.get() == config_.buffer_size:
                            # print("dropped")
                            dropped += 1
                        buffer.enqueue()
                        # print(f"enqueue - {buffer.get()}")
                    # print(f"processing 2")
                    batt.discharge(
                        time=1 * config_.timebase,
                        current=config_.board.avg_infer_energy_cons,
                    )
                    is_during_inference_semaphore -= 1
                    # model moze się zmienić w trakcie przetwarzania innego? - to się uwzględnia w is_during_inference_semaphore

                end_batt = batt.percentage()

                if config_.use_tqdm:
                    pbar.update(battery_percentage - end_batt)
                iterations += 1

        # print("***************************************************")
        # print(f"\nParticular model inference no.")
        weighted_overall_acc = 0
        inferences_no = 0
        for m in models_:
            # print(f"Model id: {m.id} - {m.get_inference_no()}")
            inferences_no += m.get_inference_no()
            weighted_overall_acc += m.get_inference_no() * m.training_accuracy
            # restart internal counters!
            m.clear_inference_no()

        inferences_no += dropped
        weighted_overall_acc /= inferences_no

        if config_.use_tqdm:
            pbar.close()

        operational_time, inferences_no, dropped, weighted_overall_acc = (
            iterations * config_.timebase,
            inferences_no - dropped,
            dropped,
            weighted_overall_acc,
        )

        if config_.print_result:
            print("***************************************************")
            print(f"mode: {config_.mode.value}")
            print(f"reference: {config_.reference}")
            print(f"min_interval: {config_.min_interval}")
            print(f"Operational time: {operational_time:.3f} ms")
            print(f"inferences_no: {inferences_no}\n")
            print(f"dropped: {dropped}\n")
            print(f"Average accuracy: {weighted_overall_acc:.2f}%\n")
            print("***************************************************")

        return operational_time, inferences_no, dropped, weighted_overall_acc


class TestSimulator(unittest.TestCase):
    # fixme: battery_capacity for tests should be 1
    def setUp(self):
        self.simulator_ = Simulator()
        self.models_usps_ = tsim.Utils.read_json("usps_tova")
        self.selectorstrategy_ = FuzzySystemHolderTova()
        self.board_ = tsim.Utils.configure_board("NUCLEO-L476RG")

    def test_sync_1(self):
        config_1 = tsim.Config.Config(
            use_tqdm_=0,
            mode_=tsim.Time.Mode.MODE_SYNC,
            min_interval_=3,
            reference_=1,
            print_result_=0,
            board_=self.board_,
        )
        # time python simulator.py --use_tqdm 1 --dataset usps_tova --mode sync --min_interval 3 --reference 1 --board NUCLEO-L476RG
        operational_time_1, inferences_no_1, dropped_1, weighted_overall_acc_1 = (
            self.simulator_.simulate(
                self.selectorstrategy_, self.models_usps_, config_1
            )
        )
        self.assertAlmostEqual(operational_time_1, 440245.630, delta=0.1)
        self.assertEqual(inferences_no_1, 146261)
        self.assertEqual(dropped_1, 0)
        self.assertAlmostEqual(weighted_overall_acc_1, 94.08, delta=0.1)
        print("OK - test_sync_1")

    def test_sync_2(self):
        config_2 = tsim.Config.Config(
            use_tqdm_=0,
            mode_=tsim.Time.Mode.MODE_SYNC,
            min_interval_=3,
            reference_=0,
            print_result_=0,
            board_=self.board_,
        )
        # time python simulator.py --use_tqdm 1 --dataset usps_tova --mode sync --min_interval 3 --reference 0 --board NUCLEO-L476RG
        operational_time_2, inferences_no_2, dropped_2, weighted_overall_acc_2 = (
            self.simulator_.simulate(
                self.selectorstrategy_, self.models_usps_, config_2
            )
        )
        self.assertAlmostEqual(operational_time_2, 587124.580, delta=0.1)
        self.assertEqual(inferences_no_2, 195058)
        self.assertEqual(dropped_2, 0)
        self.assertAlmostEqual(weighted_overall_acc_2, 93.52, delta=0.1)
        print("OK - test_sync_2")

    def test_async_1(self):
        config_3 = tsim.Config.Config(
            use_tqdm_=0,
            mode_=tsim.Time.Mode.MODE_ASYNC,
            min_interval_=2,
            reference_=1,
            print_result_=0,
            board_=self.board_,
        )
        # time python simulator.py --use_tqdm 1 --dataset usps_tova --mode async --min_interval 2 --reference 1 --board NUCLEO-L476RG
        operational_time_3, inferences_no_3, dropped_3, weighted_overall_acc_3 = (
            self.simulator_.simulate(
                self.selectorstrategy_, self.models_usps_, config_3
            )
        )
        self.assertAlmostEqual(operational_time_3, 336450.630, delta=0.1)
        self.assertEqual(inferences_no_3, 146283)
        self.assertEqual(dropped_3, 20273)
        self.assertAlmostEqual(weighted_overall_acc_3, 82.63, delta=0.1)
        print("OK - test_async_1")

    def test_async_2(self):
        config_4 = tsim.Config.Config(
            use_tqdm_=0,
            mode_=tsim.Time.Mode.MODE_ASYNC,
            min_interval_=2,
            reference_=0,
            print_result_=0,
            board_=self.board_,
        )
        # time python simulator.py --use_tqdm 1 --dataset usps_tova --mode async --min_interval 2 --reference 0 --board NUCLEO-L476RG
        operational_time, inferences_no, dropped, weighted_overall_acc = (
            self.simulator_.simulate(
                self.selectorstrategy_, self.models_usps_, config_4
            )
        )
        self.assertAlmostEqual(operational_time, 407469.580, delta=0.5)
        self.assertAlmostEqual(inferences_no, 201720, delta=10)
        self.assertEqual(dropped, 0)
        self.assertAlmostEqual(weighted_overall_acc, 93.51, delta=0.1)
        print("OK - test_async_2")

    def test_async_3(self):
        config_5 = tsim.Config.Config(
            use_tqdm_=0,
            mode_=tsim.Time.Mode.MODE_ASYNC,
            min_interval_=1,
            reference_=1,
            print_result_=0,
            board_=self.board_,
        )
        # time python simulator.py --use_tqdm 1 --dataset usps_tova --mode async --min_interval 1 --reference 1 --board NUCLEO-L476RG
        operational_time, inferences_no, dropped, weighted_overall_acc = (
            self.simulator_.simulate(
                self.selectorstrategy_, self.models_usps_, config_5
            )
        )
        self.assertAlmostEqual(operational_time, 336449.630, delta=1)
        self.assertAlmostEqual(inferences_no, 146283)
        self.assertAlmostEqual(dropped, 183574, delta=13)
        self.assertAlmostEqual(weighted_overall_acc, 41.72, delta=0.1)
        print("OK - test_async_3")

    def test_async_4(self):
        config_6 = tsim.Config.Config(
            use_tqdm_=0,
            mode_=tsim.Time.Mode.MODE_ASYNC,
            min_interval_=1,
            reference_=0,
            print_result_=0,
            board_=self.board_,
        )
        # time python simulator.py --use_tqdm 1 --dataset usps_tova --mode async --min_interval 1 --reference 0 --board NUCLEO-L476RG
        operational_time, inferences_no, dropped, weighted_overall_acc = (
            self.simulator_.simulate(
                self.selectorstrategy_, self.models_usps_, config_6
            )
        )
        self.assertAlmostEqual(operational_time, 336449.630, delta=1)
        self.assertAlmostEqual(inferences_no, 224276)
        self.assertAlmostEqual(dropped, 105567, delta=13)
        self.assertAlmostEqual(weighted_overall_acc, 63.40, delta=0.1)
        print("OK - test_async_4")

    def test_acc_first(self):
        config_7 = tsim.Config.Config(
            use_tqdm_=0,
            mode_=tsim.Time.Mode.MODE_SYNC,
            min_interval_=3,
            reference_=0,
            print_result_=0,
            board_=self.board_,
        )
        operational_time_1, inferences_no_1, dropped_1, weighted_overall_acc_1 = (
            self.simulator_.simulate(
                ReferenceSelectors(self.models_usps_, strategy="accuracy-first"),
                self.models_usps_,
                config_7,
            )
        )
        self.assertAlmostEqual(operational_time_1, 440245.630, delta=0.1)
        self.assertEqual(inferences_no_1, 146261)
        self.assertEqual(dropped_1, 0)
        self.assertAlmostEqual(weighted_overall_acc_1, 94.08, delta=0.1)
        print("OK - test_acc_first")

    def test_lat_first(self):
        config_8 = tsim.Config.Config(
            use_tqdm_=0,
            mode_=tsim.Time.Mode.MODE_SYNC,
            min_interval_=3,
            reference_=0,
            print_result_=0,
            board_=self.board_,
        )
        operational_time_1, inferences_no_1, dropped_1, weighted_overall_acc_1 = (
            self.simulator_.simulate(
                ReferenceSelectors(self.models_usps_, strategy="latency-first"),
                self.models_usps_,
                config_8,
            )
        )
        self.assertAlmostEqual(operational_time_1, 1281372.070, delta=0.1)
        self.assertEqual(inferences_no_1, 425705)
        self.assertEqual(dropped_1, 0)
        self.assertAlmostEqual(weighted_overall_acc_1, 87.50, delta=0.1)
        print("OK - test_lat_first")


# time python simulator.py --use_tqdm 1 --dataset usps_tova --mode async --min_interval 3 --reference 1 --board NUCLEO-L476RG
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Lorem ipsum...")
    parser.add_argument(
        "--use_tqdm", help="Use tqdm progress bar.", type=int, default=1, required=False
    )
    parser.add_argument("--dataset", help="Select dataset.", type=str, required=True)
    parser.add_argument("--mode", help="Select mode.", type=str, required=True)
    parser.add_argument(
        "--min_interval", help="Select minimal interval.", type=int, required=True
    )
    parser.add_argument(
        "--reference", help="Select reference.", type=int, default=1, required=False
    )
    parser.add_argument("--board", help="Select board.", type=str, required=True)

    call_args = parser.parse_args()

    board = tsim.Utils.configure_board(call_args.board)

    models = tsim.Utils.read_json(call_args.dataset)
    config = tsim.Config.Config(
        use_tqdm_=call_args.use_tqdm,
        mode_=tsim.Time.Mode(call_args.mode),
        min_interval_=call_args.min_interval,
        reference_=call_args.reference,
        print_result_=1,
        board_=board,
        battery_capacity_=100,
    )
    # selectorstrategy = FuzzySystemHolderTova()
    selectorstrategy = FuzzySystemHolderStream()
    # selectorstrategy.print_test()
    simulator = Simulator()
    simulator.simulate(selectorstrategy, models, config)

    # ####################################################

    # import threading

    # def threaded(func):
    #     def wrapper(*args, **kwargs):
    #         thread = threading.Thread(target=func, args=args)
    #         thread.start()
    #         return thread
    #     return wrapper

    # @threaded
    # def third_func(results, config):
    #     value = Simulator().simulate(FuzzySystemHolderTova(), models_usps, config)
    #     results.append(value)

    # config1 = tsim.Config.Config(
    #     use_tqdm_=call_args.use_tqdm,
    #     mode_=Mode(call_args.mode),
    #     min_interval_=call_args.min_interval,
    #     reference_=1,
    #     print_result_=0,
    # )
    # config2 = tsim.Config.Config.Config(
    #     use_tqdm_=call_args.use_tqdm,
    #     mode_=Mode(call_args.mode),
    #     min_interval_=call_args.min_interval,
    #     reference_=0,
    #     print_result_=0,
    # )

    # threads = []
    # results = []
    # for _ in range(5):
    #     threads.append(third_func(results, config1))
    #     threads.append(third_func(results, config2))

    # for thread in threads:
    #     thread.join()

    # print("------------------------")
    # print(results)
