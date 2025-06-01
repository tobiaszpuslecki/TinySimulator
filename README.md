**TinySimulator**  
*Optimizing TinyML: Balancing Energy Consumption and Inference Efficiency*  

---

## Overview

**TinySimulator** is an open-source, Python/NumPy-based simulator designed to evaluate resource-aware inference strategies in TinyML systems. Its primary goal is to help researchers and developers explore the trade-off between energy consumption, latency, accuracy, and buffer constraints when running machine-learning models on low-power microcontrollers. By simulating dynamic model (or ensemble) selection at runtime—rather than relying on a single, static TinyML model—TinySimulator enables:

- **Dynamic Model Switching**: Choose among multiple pre-profiled models (or ensembles) based on metrics such as remaining battery, buffer fill level, or desired accuracy/latency trade-off.  
- **Energy-Aware Inference**: Balance inference speed and power draw by selecting smaller, faster models when under energy pressure, and larger, more accurate models when resources allow.  
- **Realistic Workload Patterns**: Support for both *synchronous* (periodic, non-overlapping) and *asynchronous* (random inter-arrival times, with buffering) sample streams.  
- **Repeatable, Customizable Environment**: Compare different selection strategies without needing to flash hardware; tune parameters such as battery capacity, current consumption, and buffer size to match a target device.

This tool was introduced in “Optimizing TinyML: Looking for a Trade-off Between Energy Consumption and Efficiency” (Puslecki & Walkowiak), IEEE, 2025, in review.

Succesfully utilized in:
> T. Puślecki and K. Walkowiak, "A Novel One-Versus-All Approach for Multiclass Classification in TinyML Systems," in IEEE Embedded Systems Letters, vol. 17, no. 2, pp. 71-74, April 2025, doi: 10.1109/LES.2024.3482002.

---

## Key Features

1. **Dynamic Selection Module**  
   - Switches between multiple TinyML models (or ensembles) at runtime.  
   - Keeps selector latency negligible compared to model inference time.  
   - Ensures larger models do not degrade accuracy when appropriate.  
   - Meets real-time deadlines and fits within RAM/ROM constraints.

2. **Dual Sampling Patterns**  
   - **Synchronous**: Samples arrive exactly every _N_ milliseconds, no overlaps or buffering.  
   - **Asynchronous**: Randomized inter-arrival times (user-specified minimum interval + randomness), supports circular buffer; dropped samples are accounted for.

3. **Energy Modeling**  
   - Tracks battery discharge based on a simplified current-based model.  
   - Neglects microsecond-level mode-transition overhead; focuses on average current consumption in “inference” vs. “sleep” modes.  
   - Allows user to specify:  
     - Battery capacity (mAh) and initial state of charge  
     - Average current in sleep/inference  
     - Per-model inference energy (obtained via profiling tools)  

4. **Performance Metrics**  
   - **Operating Time** (device lifetime until battery exhaustion)  
   - **Number of Inferences Completed**  
   - **Dropped Samples** (asynchronous only, when buffer overflows)  
   - **Weighted Accuracy**: Accounts for dropped samples as zero; each processed sample contributes the test accuracy of its selected model.  

5. **Configurable, Extensible**  
   - Easy to integrate new selection strategies (e.g., threshold-based, accuracy-to-latency ratio, random, energy-minimizing).  
   - Parameters (buffer size, model profiles, battery specs) stored in configuration files or passed via command-line arguments.  
   - Modular architecture: swap out the energy model, sampling pattern generator, or selector logic as needed.  

---


---

## Installation

   ```bash
   git clone https://github.com/tobiaszpuslecki/TinySimulator.git
   cd TinySimulator
   python3 -m venv venv
   source venv/bin/activate   # Linux/macOS
   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Example usage

   ```bash
python simulator.py --use_tqdm 1 --dataset usps_tova --mode async --min_interval 3 --reference 1 --board NUCLEO-L476RG
   ```

## Run tests

   ```bash
   ./run_tests.sh
   ```
