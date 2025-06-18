# Assessing Resilience in Modern Energy Systems – PowerTech Tutorial 2025

## Overview 📋

This hands-on tutorial provides a complete workflow for assessing the resilience of power systems against extreme weather events. Participants will learn by modeling a realistic power network, simulating a physical hazard, and quantifying the system's operational response.

### What You’ll Learn

* Core concepts of power system modeling and resilience.
* Modeling a large-scale synthetic power grid (Texas 2000-bus system).
* Generating and visualizing spatio-temporal hazard scenarios (windstorms).
* Applying fragility models to determine component failure probabilities.
* Running Monte Carlo simulations and DC Optimal Power Flow (DC-OPF).
* Quantifying resilience using metrics like Energy Not Supplied (ENS).
* Modeling and evaluating the impact of distribution-side flexibility.

---

## Getting Started 🛠️

### ✅ Recommended Platform: [Binder](https://mybinder.org/) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/YitianDai/PowerTech2025-Tutorial.git/master)

This tutorial is designed to be run in a live, interactive environment using Binder, which requires no local installation. Binder uses the `environment.yml` file in this repository to build the environment.

#### Running the Notebooks with Binder

1.  Click the **"Launch Binder"** badge above.
2.  Binder will build the necessary environment. This may take a few minutes on the first launch.
3.  Once ready, a Jupyter interface will open in your browser.
4.  Open the notebooks in the following order, starting with `SE01`.

### Alternative: Local Jupyter Environment

If you prefer to run the notebooks locally:

#### What You Need
* A `conda` installation
* Python 3.8+

#### Local Setup
1.  Clone or download this repository.
2.  Navigate to the project directory in your terminal.
3.  Create the conda environment from the `environment.yml` file:
    ```bash
    conda env create -f environment.yml
    ```
4.  Activate the newly created environment:
    ```bash
    conda activate resilience_assessment
    ```
5.  Start Jupyter Notebook or JupyterLab:
    ```bash
    jupyter notebook
    ```

---

## Tutorial Sessions 🧠

This tutorial is divided into three main notebooks, each building on the last.

| Session | Notebook                                                    | Topic                                     |
| :------ | :---------------------------------------------------------- | :---------------------------------------- |
| **1** | `SE01_Power_Network_Model.ipynb`                            | Power System Modeling & Analysis          |
| **2** | `SE02_Hazard_Scenarios.ipynb`                               | Hazard Modeling & Vulnerability           |
| **3** | `SE03_Resilience_Assessment_with_Distribution_Flexibility.ipynb` | Resilience Quantification & Mitigation    |

---

## Learning Outcomes 🎯

By the end of this tutorial, you’ll be able to:

* Implement a full, multi-stage resilience assessment framework.
* Translate a physical hazard into component-level failure probabilities.
* Analyze the operational impact of large-scale outages.
* Evaluate the effectiveness of mitigation strategies like DS flexibility.
* Visualize complex spatio-temporal data for power systems and hazards.

---

## Prerequisites 📾

* Basic Python programming skills.
* Familiarity with libraries like `pandas` and `matplotlib`.
* Some knowledge of basic power system concepts (e.g., power flow, buses, lines).
* No prior experience with resilience assessment is required.

---

## References & Additional Resources 📚

More information about the models and some of their applications can be found at the following links:

* Dai, Y., et al. “[Whole energy system resilience vulnerability assessment](https://ieeexplore.ieee.org/document/10936044),” in *Proceeding of IET Powering Net Zero*, 2024.
* National Grid Electricity Transmission, “[Whole Energy System Resilience Vulnerability Assessment (WELLNESS)](https://smarter.energynetworks.org/projects/nget-whole-energy-system-resilience-vulnerability-assessment-sifiesrr-rd2_alpha/),” 2024.
* National Grid Electricity Transmission, “[Forward Resilience Measures](https://smarter.energynetworks.org/projects/nia_ngt0049/),” 2021.
* Birchfield, A.B., et al., "[Grid Structural Characteristics as Validation Criteria for Synthetic Networks](https://ieeexplore.ieee.org/document/7725528)," *IEEE Transactions on Power Systems*, 2017.
* [ACTIVSg2000 network dataset](https://electricgrids.engr.tamu.edu/electric-grid-test-cases/activsg2000/).