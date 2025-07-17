## Thermal Processing Profile Optimizer (ThermoPro)

The **Thermal Processing Profile Optimizer (ThermoPro)** is a machine learning-powered system designed to predict optimal drying temperature profiles and productivity based on various input process parameters. It also incorporates a crucial **Failure Mode and Effects Analysis (FMEA)** module to assess and mitigate potential risks associated with the recommended thermal profiles, even offering productivity adjustments and re-analysis of FMEA to ensure a robust and safe drying process.

## ✨ Features

  * **Intelligent Dryer Type Classification**: Automatically recommends the most suitable dryer type (Batch or Belt) based on your material and process requirements.
  * **Optimal Thermal Parameter Prediction**: Predicts key parameters like heating rate, cooling rate, hold time factor, and peak temperature adjustment for the chosen dryer.
  * **Productivity Estimation**: Provides an estimated productivity (units/hour) for the predicted profile.
  * **Dynamic Thermal Profile Generation**: Generates a detailed time-temperature thermal profile for visualization and analysis.
  * **User-Adjustable Productivity**: Allows users to specify a desired productivity, and the system intelligently adjusts thermal parameters to meet this target while re-evaluating the FMEA risks.
  * **Integrated FMEA Analysis**: Performs a comprehensive FMEA to identify potential failure modes, their effects, causes, and recommended actions, complete with Risk Priority Numbers (RPN) and risk levels.
  * **Actionable FMEA Report & Plan**: Generates a structured FMEA report highlighting critical risks and an actionable plan categorized by urgency (immediate, short-term, long-term).
  * **Clear Reporting & Visualization**: Presents results in easy-to-understand reports and offers a graphical representation of the thermal profile.
  * **Persistent Models**: Saves trained machine learning models, so you don't need to retrain them every time.

## 🚀 Getting Started

### Prerequisites

  * Python 3.x
  * `scikit-learn`
  * `numpy`
  * `matplotlib`
  * `joblib`

You can install the necessary libraries using pip:

```bash
pip install scikit-learn numpy matplotlib joblib
```

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/thermopro.git
    cd thermopro
    ```
2.  **Run the application:**
    ```bash
    python your_main_script_name.py
    ```
    (Replace `your_main_script_name.py` with the actual name of your Python file, likely `main.py` if you've packaged it as such, or directly the file containing the `ThermalProcessingProfileOptimizer` class and `main` function.)

### Usage

Upon running the script, the system will first check for pre-trained models. If they don't exist, it will train them using synthetic data. This might take a moment.

```
THERMOPRO - Thermal Processing & Productivity Optimizer
============================================================
Models not found; training new models...
Generating synthetic data...
Training dryer type classifier...
Accuracy: 0.945
Cross-validation: 0.947
              precision    recall  f1-score   avg/total
 Batch_Dryer       0.93      0.97      0.95       0.95
   Belt_Dryer      0.96      0.92      0.94       0.95

Training regression models for Batch_Dryer...
  heating_rate: MSE=0.088, R2=0.916
  cooling_rate: MSE=0.038, R2=0.884
  hold_time_factor: MSE=0.010, R2=0.918
  peak_temp_adjustment: MSE=3.992, R2=0.897
  productivity: MSE=23.364, R2=0.914
Training regression models for Belt_Dryer...
  heating_rate: MSE=0.040, R2=0.926
  cooling_rate: MSE=0.019, R2=0.902
  hold_time_factor: MSE=0.011, R2=0.906
  peak_temp_adjustment: MSE=2.274, R2=0.900
  productivity: MSE=23.630, R2=0.903
All models trained and saved successfully.

============================================================
THERMAL PARAMETER INPUT
============================================================
Enter target temperature (20–200°C): 120
Enter exposure time (10–600 min): 90
Enter material thickness (0.1–50.0 mm): 5.0

============================================================
THERMAL PROCESSING & PRODUCTIVITY REPORT
============================================================
    Target Temperature: 120.0 °C
    Exposure Time: 90.0 minutes
    Material Thickness: 5.00 mm
    Recommended Dryer: Batch_Dryer (80.1%)

    Heating Rate: 3.55 °C/min
    Cooling Rate: 2.37 °C/min
    Hold Time Factor: 1.16
    Peak Temperature Adjustment: 10.9 °C
    Productivity: 57.0 units/hour

    Peak Temperature: 130.9 °C
    Total Cycle Time: 130.6 minutes
    Heating Time: 31.2 minutes
    Holding Time: 104.4 minutes
    Cooling Time: 54.4 minutes

Safety & Control Recommendations:
    Heating Rate: Risk: Thermal shock or material degradation if rate is too high; Control: Monitor temperature gradients and ensure uniform heating; Range: 0.5–10.0 °C/min
    Cooling Rate: Risk: Cracking or poor solidification if cooling too rapid; Control: Control cooling environment and monitor gradients; Range: 0.3–8.0 °C/min
    Hold Time: Risk: Incomplete drying or energy waste if hold time incorrect; Control: Monitor completion and optimize duration; Range: 30–300% of exposure time
    Peak Temperature: Risk: Damage or safety hazard if temperature out of range; Control: Use interlocks and real-time monitoring; Range: Target ±20°C
    Productivity: Risk: Bottleneck if throughput too low; Control: Balance speed and quality monitoring; Range: Context-dependent

Enter your desired productivity (units/hour): 70

Adjusted Parameters to achieve desired productivity:
    heating_rate: 4.36
    cooling_rate: 2.92
    hold_time_factor: 1.43
    peak_temp_adjustment: 10.93
    productivity: 70.00

Performing FMEA on adjusted parameters...

================================================================================
FMEA RISK ANALYSIS REPORT
================================================================================

Risk Level Distribution:
    Medium: 2
    High: 2
    Low: 1
    Very Low: 1

Key Failure Modes (RPN ≥ 50):
--------------------------------------------------------------------------------
1. Mode: Overheating | Phase: heating_phase
    Effects: ['Material damage', 'Fire risk', 'Energy waste']
    Causes: ['Heater control failure', 'Sensor malfunction', 'Cooling system fault']
    S=10 O=5 D=3 RPN=150 (High)
    Controls: ['Temperature monitoring', 'Safety interlock', 'Alarm system']
    Actions: ['Regular sensor calibration', 'Dual safety system', 'Automatic shutdown']

2. Mode: Rapid Cooling | Phase: cooling_phase
    Effects: ['Thermal shock', 'Cracking', 'Poor quality']
    Causes: ['Overactive cooling', 'Ambient drop', 'Control fault']
    S=8 O=4 D=4 RPN=128 (High)
    Controls: ['Rate control', 'Temperature monitoring']
    Actions: ['Limit rate', 'Stage cooling', 'Control ambient']

3. Mode: Temperature Gradient | Phase: heating_phase
    Effects: ['Local overheating', 'Material deformation', 'Inconsistent quality']
    Causes: ['Uneven heat distribution', 'Poor insulation', 'Blocked airflow']
    S=7 O=7 D=5 RPN=245 (Medium)
    Controls: ['Multipoint sensors', 'Air circulation system']
    Actions: ['Enhance distribution', 'Inspect insulation', 'Optimize airflow']

4. Mode: Temperature Maintenance Failure | Phase: holding_phase
    Effects: ['Incomplete process', 'Quality loss', 'Energy waste']
    Causes: ['Controller failure', 'Heat loss', 'Load change']
    S=8 O=3 D=2 RPN=48 (Medium)
    Controls: ['Control system', 'Insulation checks']
    Actions: ['Controller redundancy', 'Improve insulation', 'Stabilize load']

Complete FMEA Table:
No.  Mode                       RPN   Level      S   O   D
--------------------------------------------------------------------------------
1    Temperature Gradient       245   Medium     7   7   5
2    Overheating                150   High       10  5   3
3    Rapid Cooling              128   High       8   4   4
4    Temperature Maintenance    48    Medium     8   3   2
5    Residence Time Deviation   45    Low        5   3   3
6    Insufficient Cooling       30    Very Low   6   2   3

================================================================================
FMEA-BASED ACTION PLAN
================================================================================

Immediate Actions (RPN ≥ 200):

Short-Term Actions (100 ≤ RPN < 200):
    1. [Overheating] Regular sensor calibration
    2. [Overheating] Dual safety system
    3. [Overheating] Automatic shutdown
    4. [Rapid Cooling] Limit rate
    5. [Rapid Cooling] Stage cooling
    6. [Rapid Cooling] Control ambient

Long-Term Actions (50 ≤ RPN < 100):
    1. [Temperature Gradient] Enhance distribution
    2. [Temperature Gradient] Inspect insulation
    3. [Temperature Gradient] Optimize airflow

Monitoring Points:
    1. Overheating (RPN=150)
    2. Rapid Cooling (RPN=128)
    3. Temperature Gradient (RPN=245)

Performing FMEA on original parameters...

================================================================================
FMEA RISK ANALYSIS REPORT
================================================================================

Risk Level Distribution:
    Medium: 2
    High: 1
    Very High: 1
    Low: 1
    Very Low: 1

Key Failure Modes (RPN ≥ 50):
--------------------------------------------------------------------------------
1. Mode: Overheating | Phase: heating_phase
    Effects: ['Material damage', 'Fire risk', 'Energy waste']
    Causes: ['Heater control failure', 'Sensor malfunction', 'Cooling system fault']
    S=9 O=4 D=3 RPN=108 (High)
    Controls: ['Temperature monitoring', 'Safety interlock', 'Alarm system']
    Actions: ['Regular sensor calibration', 'Dual safety system', 'Automatic shutdown']

2. Mode: Temperature Gradient | Phase: heating_phase
    Effects: ['Local overheating', 'Material deformation', 'Inconsistent quality']
    Causes: ['Uneven heat distribution', 'Poor insulation', 'Blocked airflow']
    S=7 O=6 D=5 RPN=210 (Very High)
    Controls: ['Multipoint sensors', 'Air circulation system']
    Actions: ['Enhance distribution', 'Inspect insulation', 'Optimize airflow']

3. Mode: Temperature Maintenance Failure | Phase: holding_phase
    Effects: ['Incomplete process', 'Quality loss', 'Energy waste']
    Causes: ['Controller failure', 'Heat loss', 'Load change']
    S=8 O=3 D=2 RPN=48 (Medium)
    Controls: ['Control system', 'Insulation checks']
    Actions: ['Controller redundancy', 'Improve insulation', 'Stabilize load']

Complete FMEA Table:
No.  Mode                       RPN   Level      S   O   D
--------------------------------------------------------------------------------
1    Temperature Gradient       210   Very High  7   6   5
2    Overheating                108   High       9   4   3
3    Temperature Maintenance    48    Medium     8   3   2
4    Residence Time Deviation   45    Low        5   3   3
5    Rapid Cooling              32    Low        8   1   4
6    Insufficient Cooling       30    Very Low   6   2   3

================================================================================
FMEA-BASED ACTION PLAN
================================================================================

Immediate Actions (RPN ≥ 200):
    1. [Temperature Gradient] Enhance distribution
    2. [Temperature Gradient] Inspect insulation
    3. [Temperature Gradient] Optimize airflow

Short-Term Actions (100 ≤ RPN < 200):
    1. [Overheating] Regular sensor calibration
    2. [Overheating] Dual safety system
    3. [Overheating] Automatic shutdown

Long-Term Actions (50 ≤ RPN < 100):

Monitoring Points:
    1. Overheating (RPN=108)
    2. Temperature Gradient (RPN=210)

Show profile plot? (y/n): y
```

Follow the prompts to enter your desired `target_temperature`, `exposure_time`, and `material_thickness`. The system will then output a detailed report, including the recommended dryer type, thermal parameters, and a comprehensive FMEA analysis. You'll also have the option to view a plot of the generated thermal profile.

## ⚙️ How it Works

The `ThermalProcessingProfileOptimizer` class encapsulates the entire workflow:

1.  **Data Generation**: `generate_synthetic_data` and `generate_parameter_targets` create realistic simulated data for training.
2.  **Model Training**:
      * A **Random Forest Classifier** is trained to predict the optimal dryer type (Batch or Belt) based on input features.
      * Separate **Linear Regression** models are trained for each dryer type and each thermal parameter (heating rate, cooling rate, hold time factor, peak temperature adjustment, and productivity).
3.  **Model Persistence**: Trained models (classifier, scaler, encoder, and regression models) are saved to disk using `joblib` for reusability.
4.  **Prediction**: The `predict_thermal_profile` method takes new input parameters, scales them, predicts the dryer type, and then uses the appropriate regression models to predict the thermal parameters and productivity.
5.  **Thermal Profile Generation**: `generate_thermal_profile` constructs a time-temperature profile based on the predicted thermal parameters.
6.  **Productivity Adjustment**: The `adjust_for_desired_productivity` function intelligently modifies the heating rate, cooling rate, and hold time factor to meet a user-specified productivity target.
7.  **FMEA Analysis**:
      * The `initialize_fmea_database` sets up a predefined database of common failure modes, their effects, causes, current controls, and recommended actions.
      * `perform_fmea_analysis` calculates the **Risk Priority Number (RPN)** ($RPN = \\text{Severity} \\times \\text{Occurrence} \\times \\text{Detection}$) for each failure mode, adjusting occurrence and severity based on the predicted thermal parameters (e.g., higher heating rates might increase occurrence of "Overheating").
      * `get_risk_level` classifies the risk based on the RPN.
      * `generate_fmea_action_plan` creates a prioritized action plan based on the RPNs.
8.  **Reporting & Plotting**: `print_thermal_report`, `print_fmea_report`, `print_action_plan`, and `plot_thermal_profile` provide comprehensive output and visualizations.

```

## 🤝 Contributing

Contributions are welcome\! If you have suggestions for improvements, new features, or bug fixes, please open an issue or submit a pull request.

## 📄 License

This project is open-source and available under the [MIT License](https://www.google.com/search?q=LICENSE).

-----
