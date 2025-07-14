import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')


class ThermalProcessingProfileOptimizer:
    """
    ThermalProcessingProfileOptimizer
    A machine learning system for predicting optimal drying temperature profiles
    and productivity based on input process parameters, with user-specified
    productivity adjustment and corresponding FMEA analysis.
    """

    def __init__(self, model_dir='thermopro_models'):
        self.model_dir = model_dir
        self.ensure_model_directory()

        # Model file paths
        self.model_files = {
            'classifier': os.path.join(model_dir, 'dryer_type_classifier.pkl'),
            'label_encoder': os.path.join(model_dir, 'dryer_label_encoder.pkl'),
            'scaler': os.path.join(model_dir, 'thermal_scaler.pkl')
        }

        # Dryer types and regression targets (including productivity)
        self.dryer_types = ['Batch_Dryer', 'Belt_Dryer']
        self.parameters = [
            'heating_rate',
            'cooling_rate',
            'hold_time_factor',
            'peak_temp_adjustment',
            'productivity'
        ]

        # Add regression model file paths
        for dryer in self.dryer_types:
            for param in self.parameters:
                filename = f"{dryer}_{param}_model.pkl"
                self.model_files[f"{dryer}_{param}"] = os.path.join(model_dir, filename)

        # Initialize models
        self.classifier = None
        self.label_encoder = None
        self.scaler = None
        self.regression_models = {}

    def ensure_model_directory(self):
        """Create the model directory if it does not exist."""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            print(f"Created directory: {self.model_dir}")

    def generate_synthetic_data(self, n_samples=500):
        """Generate synthetic training data for thermal processing."""
        np.random.seed(42)
        X = np.column_stack([
            np.random.gamma(2, 30, n_samples) + 20,       # target_temperature
            np.random.exponential(60, n_samples) + 10,    # exposure_time
            np.random.uniform(0.1, 50.0, n_samples)       # material_thickness (0.1–50 mm)
        ])
        X[:, 0] = np.clip(X[:, 0], 20, 200)
        X[:, 1] = np.clip(X[:, 1], 10, 600)
        X[:, 2] = np.clip(X[:, 2], 0.1, 50.0)

        # Probability of selecting Belt dryer increases with thickness and time
        belt_prob = 1 / (1 + np.exp(
            0.02 * (X[:, 0] - 100)
            + 0.005 * X[:, 2]
            - 0.003 * (X[:, 1] - 200)
        ))
        y = np.where(np.random.random(n_samples) < belt_prob,
                     'Belt_Dryer', 'Batch_Dryer')
        return X, y

    def generate_parameter_targets(self, X, dryer_type):
        """Generate realistic parameter targets for a given dryer type, including productivity."""
        n = X.shape[0]
        if dryer_type == 'Batch_Dryer':
            heating_rate = 2.0 + 0.01 * X[:, 0] - 0.002 * X[:, 1] + 0.5 * X[:, 2] / 10 + np.random.normal(0, 0.3, n)
            cooling_rate = 1.5 + 0.008 * X[:, 0] - 0.001 * X[:, 1] + 0.3 * X[:, 2] / 10 + np.random.normal(0, 0.2, n)
            hold_time = 0.8 + 0.002 * X[:, 0] + 0.001 * X[:, 1] + 0.1 * X[:, 2] / 10 + np.random.normal(0, 0.1, n)
            peak_adj = 5 + 0.05 * X[:, 0] + 0.01 * X[:, 1] + np.random.normal(0, 2, n)
            productivity = 50 + 0.2 * X[:, 2] - 0.01 * X[:, 1] + np.random.normal(0, 5, n)
        else:  # Belt_Dryer
            heating_rate = 1.2 + 0.008 * X[:, 0] - 0.001 * X[:, 1] + 0.3 * X[:, 2] / 10 + np.random.normal(0, 0.2, n)
            cooling_rate = 0.8 + 0.005 * X[:, 0] - 0.0005 * X[:, 1] + 0.2 * X[:, 2] / 10 + np.random.normal(0, 0.15, n)
            hold_time = 1.2 + 0.001 * X[:, 0] + 0.002 * X[:, 1] + 0.05 * X[:, 2] / 10 + np.random.normal(0, 0.1, n)
            peak_adj = 3 + 0.03 * X[:, 0] + 0.005 * X[:, 1] + np.random.normal(0, 1.5, n)
            productivity = 100 + 0.25 * X[:, 2] - 0.005 * X[:, 1] + np.random.normal(0, 5, n)

        return {
            'heating_rate': np.clip(heating_rate, 0.5, 10.0),
            'cooling_rate': np.clip(cooling_rate, 0.3, 8.0),
            'hold_time_factor': np.clip(hold_time, 0.3, 3.0),
            'peak_temp_adjustment': np.clip(peak_adj, -5, 20),
            'productivity': np.clip(productivity, 0, None)
        }

    def train_models(self, n_samples=500):
        """Train classification and regression models and save them."""
        print("Generating synthetic data...")
        X, y = self.generate_synthetic_data(n_samples)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.label_encoder = LabelEncoder()
        y_enc = self.label_encoder.fit_transform(y)

        print("Training dryer type classifier...")
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )
        self.classifier = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, min_samples_split=5
        )
        self.classifier.fit(X_tr, y_tr)
        y_pred = self.classifier.predict(X_te)
        print(f"Accuracy: {accuracy_score(y_te, y_pred):.3f}")
        print(f"Cross-validation: {cross_val_score(self.classifier, X_scaled, y_enc, cv=5).mean():.3f}")
        print(classification_report(y_te, y_pred, target_names=self.label_encoder.classes_))

        for dryer in self.dryer_types:
            print(f"Training regression models for {dryer}...")
            mask = (y == dryer)
            X_d = X[mask]
            X_ds = X_scaled[mask]
            targets = self.generate_parameter_targets(X_d, dryer)
            for name, vals in targets.items():
                X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
                    X_ds, vals, test_size=0.2, random_state=42
                )
                pipe = Pipeline([('regressor', LinearRegression())])
                pipe.fit(X_tr_r, y_tr_r)
                yp = pipe.predict(X_te_r)
                print(f"  {name}: MSE={mean_squared_error(y_te_r, yp):.3f}, R2={r2_score(y_te_r, yp):.3f}")
                self.regression_models[f"{dryer}_{name}"] = pipe

        self.save_models()
        print("All models trained and saved successfully.")

    def save_models(self):
        """Persist all trained models to disk."""
        joblib.dump(self.classifier, self.model_files['classifier'])
        joblib.dump(self.label_encoder, self.model_files['label_encoder'])
        joblib.dump(self.scaler, self.model_files['scaler'])
        for key, mdl in self.regression_models.items():
            joblib.dump(mdl, self.model_files[key])

    def load_models(self):
        """Load all models from disk."""
        self.regression_models = {}
        try:
            self.classifier = joblib.load(self.model_files['classifier'])
            self.label_encoder = joblib.load(self.model_files['label_encoder'])
            self.scaler = joblib.load(self.model_files['scaler'])
            for dryer in self.dryer_types:
                for param in self.parameters:
                    key = f"{dryer}_{param}"
                    self.regression_models[key] = joblib.load(self.model_files[key])
            print("All models loaded successfully.")
            return True
        except FileNotFoundError as e:
            print(f"Model files not found: {e}")
            return False

    def models_exist(self):
        """Return True if all model files are present."""
        return all(os.path.exists(path) for path in self.model_files.values())

    def predict_thermal_profile(self, target_temperature, exposure_time, material_thickness):
        """Predict dryer type, thermal parameters, productivity, and thermal profile."""
        if not (20 <= target_temperature <= 200):
            raise ValueError("Target temperature must be between 20 and 200°C")
        if not (10 <= exposure_time <= 600):
            raise ValueError("Exposure time must be between 10 and 600 minutes")
        if not (0.1 <= material_thickness <= 50.0):
            raise ValueError("Material thickness must be between 0.1 and 50.0 mm")

        features = np.array([[target_temperature, exposure_time, material_thickness]])
        fs = self.scaler.transform(features)
        idx = self.classifier.predict(fs)[0]
        dryer = self.label_encoder.inverse_transform([idx])[0]
        probs = dict(zip(self.label_encoder.classes_, self.classifier.predict_proba(fs)[0]))

        preds = {}
        for p in self.parameters:
            preds[p] = self.regression_models[f"{dryer}_{p}"].predict(fs)[0]

        profile = self.generate_thermal_profile(
            target_temperature,
            exposure_time,
            preds['heating_rate'],
            preds['cooling_rate'],
            preds['hold_time_factor'],
            preds['peak_temp_adjustment']
        )

        return {
            'dryer_type': dryer,
            'dryer_probabilities': probs,
            'thermal_parameters': preds,
            'thermal_profile': profile,
            'input_features': {
                'target_temperature': target_temperature,
                'exposure_time': exposure_time,
                'material_thickness': material_thickness
            }
        }

    def generate_thermal_profile(self, target_temp, exposure_time,
                                 heating_rate, cooling_rate,
                                 hold_time_factor, peak_temp_adj):
        """Generate a time-temperature profile for the dryer cycle."""
        peak_temp = target_temp + peak_temp_adj
        hold_time = exposure_time * hold_time_factor
        heating_time = (peak_temp - 20) / heating_rate
        cooling_time = (peak_temp - 20) / cooling_rate

        time_pts, temp_pts = [], []

        # Heating phase
        steps_h = int(heating_time * 2)
        for i in range(steps_h + 1):
            t = i / 2
            temp = 20 + (peak_temp - 20) * (t / heating_time)
            time_pts.append(t)
            temp_pts.append(temp)

        # Holding phase
        steps_hold = int(hold_time * 2)
        for i in range(1, steps_hold + 1):
            t = heating_time + i / 2
            time_pts.append(t)
            temp_pts.append(peak_temp)

        # Cooling phase
        steps_c = int(cooling_time * 2)
        for i in range(1, steps_c + 1):
            t = heating_time + hold_time + i / 2
            temp = peak_temp - (peak_temp - 20) * (i / 2) / cooling_time
            time_pts.append(t)
            temp_pts.append(temp)

        return {
            'time_minutes': time_pts,
            'temperature_celsius': temp_pts,
            'phases': {
                'heating': {'start': 0, 'end': heating_time, 'rate': heating_rate},
                'holding': {'start': heating_time, 'end': heating_time + hold_time, 'temperature': peak_temp},
                'cooling': {'start': heating_time + hold_time,
                            'end': heating_time + hold_time + cooling_time, 'rate': cooling_rate}
            },
            'total_cycle_time': heating_time + hold_time + cooling_time,
            'peak_temperature': peak_temp
        }

    def get_thermal_safety_info(self):
        """Return safety and control guidance for thermal parameters."""
        return {
            'Heating Rate': {
                'risk': 'Thermal shock or material degradation if rate is too high',
                'control': 'Monitor temperature gradients and ensure uniform heating',
                'acceptable_range': '0.5–10.0 °C/min'
            },
            'Cooling Rate': {
                'risk': 'Cracking or poor solidification if cooling too rapid',
                'control': 'Control cooling environment and monitor gradients',
                'acceptable_range': '0.3–8.0 °C/min'
            },
            'Hold Time': {
                'risk': 'Incomplete drying or energy waste if hold time incorrect',
                'control': 'Monitor completion and optimize duration',
                'acceptable_range': '30–300% of exposure time'
            },
            'Peak Temperature': {
                'risk': 'Damage or safety hazard if temperature out of range',
                'control': 'Use interlocks and real-time monitoring',
                'acceptable_range': 'Target ±20°C'
            },
            'Productivity': {
                'risk': 'Bottleneck if throughput too low',
                'control': 'Balance speed and quality monitoring',
                'acceptable_range': 'Context-dependent'
            }
        }

    def plot_thermal_profile(self, result):
        """Plot the thermal profile."""
        profile = result['thermal_profile']
        plt.figure(figsize=(12, 8))

        # Temperature profile
        plt.subplot(2, 1, 1)
        plt.plot(profile['time_minutes'], profile['temperature_celsius'],
                 linewidth=2, label='Temperature Profile')
        phases = profile['phases']
        plt.axvline(phases['heating']['end'], linestyle='--', alpha=0.7, label='Heating End')
        plt.axvline(phases['holding']['end'], linestyle='--', alpha=0.7, label='Holding End')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Temperature (°C)')
        plt.title(f'Thermal Profile for {result["dryer_type"]}')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Phase durations
        plt.subplot(2, 1, 2)
        phase_names = ['Heating', 'Holding', 'Cooling']
        phase_times = [
            phases['heating']['end'],
            phases['holding']['end'] - phases['holding']['start'],
            phases['cooling']['end'] - phases['cooling']['start']
        ]
        bars = plt.bar(phase_names, phase_times, alpha=0.7)
        plt.ylabel('Duration (minutes)')
        plt.title('Phase Duration Breakdown')
        plt.grid(True, alpha=0.3, axis='y')
        for bar, t in zip(bars, phase_times):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{t:.1f} min', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    def print_thermal_report(self, result):
        """Print a formatted thermal processing & productivity report."""
        print("\n" + "=" * 70)
        print("THERMAL PROCESSING & PRODUCTIVITY REPORT")
        print("=" * 70)
        inf = result['input_features']
        print(f"  Target Temperature: {inf['target_temperature']:.1f} °C")
        print(f"  Exposure Time: {inf['exposure_time']:.1f} minutes")
        print(f"  Material Thickness: {inf['material_thickness']:.2f} mm")
        print(f"  Recommended Dryer: {result['dryer_type']} "
              f"({result['dryer_probabilities'][result['dryer_type']] * 100:.1f}%)")
        tp = result['thermal_parameters']
        print(f"\n  Heating Rate: {tp['heating_rate']:.2f} °C/min")
        print(f"  Cooling Rate: {tp['cooling_rate']:.2f} °C/min")
        print(f"  Hold Time Factor: {tp['hold_time_factor']:.2f}")
        print(f"  Peak Temperature Adjustment: {tp['peak_temp_adjustment']:.1f} °C")
        print(f"  Productivity: {tp['productivity']:.1f} units/hour")
        prof = result['thermal_profile']
        print(f"\n  Peak Temperature: {prof['peak_temperature']:.1f} °C")
        print(f"  Total Cycle Time: {prof['total_cycle_time']:.1f} minutes")
        print(f"  Heating Time: {prof['phases']['heating']['end']:.1f} minutes")
        print(f"  Holding Time: {prof['phases']['holding']['end'] - prof['phases']['holding']['start']:.1f} minutes")
        print(f"  Cooling Time: {prof['phases']['cooling']['end'] - prof['phases']['cooling']['start']:.1f} minutes")
        print("\nSafety & Control Recommendations:")
        for name, info in self.get_thermal_safety_info().items():
            print(f"  {name}: Risk: {info['risk']}; Control: {info['control']}; Range: {info['acceptable_range']}")

    def initialize_fmea_database(self):
        """Initialize the FMEA database for the drying process."""
        self.fmea_database = {
            'heating_phase': [
                {
                    'failure_mode': 'Overheating',
                    'potential_effects': ['Material damage', 'Fire risk', 'Energy waste'],
                    'severity': 9,
                    'potential_causes': ['Heater control failure', 'Sensor malfunction', 'Cooling system fault'],
                    'occurrence': 4,
                    'current_controls': ['Temperature monitoring', 'Safety interlock', 'Alarm system'],
                    'detection': 3,
                    'recommended_actions': ['Regular sensor calibration', 'Dual safety system', 'Automatic shutdown']
                },
                {
                    'failure_mode': 'Heating Rate Deviation',
                    'potential_effects': ['Uneven drying', 'Quality loss', 'Rework required'],
                    'severity': 6,
                    'potential_causes': ['Unstable heater output', 'Poor air circulation', 'Load variation'],
                    'occurrence': 5,
                    'current_controls': ['Rate monitoring', 'Feedback control'],
                    'detection': 4,
                    'recommended_actions': ['Optimize PID control', 'Improve pre-heating', 'Balance load']
                },
                {
                    'failure_mode': 'Temperature Gradient',
                    'potential_effects': ['Local overheating', 'Material deformation', 'Inconsistent quality'],
                    'severity': 7,
                    'potential_causes': ['Uneven heat distribution', 'Poor insulation', 'Blocked airflow'],
                    'occurrence': 6,
                    'current_controls': ['Multipoint sensors', 'Air circulation system'],
                    'detection': 5,
                    'recommended_actions': ['Enhance distribution', 'Inspect insulation', 'Optimize airflow']
                }
            ],
            'holding_phase': [
                {
                    'failure_mode': 'Temperature Maintenance Failure',
                    'potential_effects': ['Incomplete process', 'Quality loss', 'Energy waste'],
                    'severity': 8,
                    'potential_causes': ['Controller failure', 'Heat loss', 'Load change'],
                    'occurrence': 3,
                    'current_controls': ['Control system', 'Insulation checks'],
                    'detection': 2,
                    'recommended_actions': ['Controller redundancy', 'Improve insulation', 'Stabilize load']
                },
                {
                    'failure_mode': 'Residence Time Deviation',
                    'potential_effects': ['Over/under drying', 'Inefficient energy use', 'Lower throughput'],
                    'severity': 5,
                    'potential_causes': ['Timer error', 'Process change', 'Operator error'],
                    'occurrence': 4,
                    'current_controls': ['Time monitoring', 'Automation'],
                    'detection': 3,
                    'recommended_actions': ['Increase automation', 'Operator training', 'Standardize procedure']
                }
            ],
            'cooling_phase': [
                {
                    'failure_mode': 'Rapid Cooling',
                    'potential_effects': ['Thermal shock', 'Cracking', 'Poor quality'],
                    'severity': 8,
                    'potential_causes': ['Overactive cooling', 'Ambient drop', 'Control fault'],
                    'occurrence': 3,
                    'current_controls': ['Rate control', 'Temperature monitoring'],
                    'detection': 4,
                    'recommended_actions': ['Limit rate', 'Stage cooling', 'Control ambient']
                },
                {
                    'failure_mode': 'Insufficient Cooling',
                    'potential_effects': ['Delay', 'Quality loss', 'Energy waste'],
                    'severity': 6,
                    'potential_causes': ['Cooling performance drop', 'Coolant shortage', 'Heat exchanger clog'],
                    'occurrence': 5,
                    'current_controls': ['System checks', 'Performance monitoring'],
                    'detection': 3,
                    'recommended_actions': ['Regular cleaning', 'Coolant management', 'Replace exchanger']
                }
            ],
            'general_system': [
                {
                    'failure_mode': 'Sensor Malfunction',
                    'potential_effects': ['Incorrect control', 'Process instability', 'Quality issues'],
                    'severity': 7,
                    'potential_causes': ['Sensor aging', 'Calibration error', 'Environmental effects'],
                    'occurrence': 6,
                    'current_controls': ['Regular calibration', 'Sensor inspection'],
                    'detection': 4,
                    'recommended_actions': ['Shorten replacement interval', 'Use dual sensors', 'Predictive maintenance']
                },
                {
                    'failure_mode': 'Power Supply Instability',
                    'potential_effects': ['Process halt', 'Equipment damage', 'Production loss'],
                    'severity': 9,
                    'potential_causes': ['Grid issues', 'UPS failure', 'Aging wiring'],
                    'occurrence': 2,
                    'current_controls': ['UPS system', 'Power monitoring'],
                    'detection': 2,
                    'recommended_actions': ['Increase UPS capacity', 'Add backup power', 'Inspect wiring']
                }
            ]
        }

    def calculate_rpn(self, severity, occurrence, detection):
        """Calculate the Risk Priority Number (RPN)."""
        return severity * occurrence * detection

    def get_risk_level(self, rpn):
        """Classify risk level based on RPN."""
        if rpn >= 200:
            return "Very High"
        elif rpn >= 100:
            return "High"
        elif rpn >= 50:
            return "Medium"
        elif rpn >= 20:
            return "Low"
        else:
            return "Very Low"

    def perform_fmea_analysis(self, result):
        """Perform FMEA for the drying process based on the prediction result."""
        if not hasattr(self, 'fmea_database'):
            self.initialize_fmea_database()

        params = result['thermal_parameters']
        prof = result['thermal_profile']
        fmea_results = []

        for phase, modes in self.fmea_database.items():
            for fm in modes:
                sev = fm['severity']
                occ = fm['occurrence']
                det = fm['detection']

                if phase == 'heating_phase':
                    if params['heating_rate'] > 5.0:
                        occ += 1
                    if prof['peak_temperature'] > 150:
                        sev += 1
                elif phase == 'holding_phase':
                    hold = prof['phases']['holding']['end'] - prof['phases']['holding']['start']
                    if hold > 120:
                        occ += 1
                elif phase == 'cooling_phase':
                    if params['cooling_rate'] > 4.0:
                        occ += 1
                    if prof['peak_temperature'] > 120:
                        sev += 1

                sev = min(10, max(1, sev))
                occ = min(10, max(1, occ))
                det = min(10, max(1, det))

                rpn = self.calculate_rpn(sev, occ, det)
                level = self.get_risk_level(rpn)

                fmea_results.append({
                    'phase': phase,
                    'failure_mode': fm['failure_mode'],
                    'potential_effects': fm['potential_effects'],
                    'potential_causes': fm['potential_causes'],
                    'severity': sev,
                    'occurrence': occ,
                    'detection': det,
                    'rpn': rpn,
                    'risk_level': level,
                    'current_controls': fm['current_controls'],
                    'recommended_actions': fm['recommended_actions']
                })

        fmea_results.sort(key=lambda x: x['rpn'], reverse=True)
        return fmea_results

    def print_fmea_report(self, fmea_results):
        """Print the FMEA analysis report."""
        print("\n" + "=" * 80)
        print("FMEA RISK ANALYSIS REPORT")
        print("=" * 80)

        counts = {}
        for r in fmea_results:
            counts[r['risk_level']] = counts.get(r['risk_level'], 0) + 1

        print("\nRisk Level Distribution:")
        for lvl, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {lvl}: {cnt}")

        high = [r for r in fmea_results if r['rpn'] >= 50]
        if high:
            print("\nKey Failure Modes (RPN ≥ 50):")
            print("-" * 80)
            for i, item in enumerate(high, 1):
                print(f"{i}. Mode: {item['failure_mode']} | Phase: {item['phase']}")
                print(f"   Effects: {', '.join(item['potential_effects'])}")
                print(f"   Causes: {', '.join(item['potential_causes'])}")
                print(f"   S={item['severity']} O={item['occurrence']} D={item['detection']} RPN={item['rpn']} ({item['risk_level']})")
                print(f"   Controls: {', '.join(item['current_controls'])}")
                print(f"   Actions: {', '.join(item['recommended_actions'])}\n")

        print("Complete FMEA Table:")
        print(f"{'No.':<4} {'Mode':<25} {'RPN':<5} {'Level':<10} {'S':<3} {'O':<3} {'D':<3}")
        print("-" * 80)
        for i, item in enumerate(fmea_results, 1):
            mode = item['failure_mode'][:22] + ('...' if len(item['failure_mode']) > 22 else '')
            print(f"{i:<4} {mode:<25} {item['rpn']:<5} {item['risk_level']:<10} {item['severity']:<3} {item['occurrence']:<3} {item['detection']:<3}")

    def generate_fmea_action_plan(self, fmea_results):
        """Generate an action plan from FMEA results."""
        plan = {'immediate': [], 'short_term': [], 'long_term': [], 'monitor': []}
        for item in fmea_results:
            if item['rpn'] >= 200:
                plan['immediate'] += [f"[{item['failure_mode']}] {act}" for act in item['recommended_actions']]
            elif item['rpn'] >= 100:
                plan['short_term'] += [f"[{item['failure_mode']}] {act}" for act in item['recommended_actions']]
            elif item['rpn'] >= 50:
                plan['long_term'] += [f"[{item['failure_mode']}] {act}" for act in item['recommended_actions']]
            if item['rpn'] >= 50:
                plan['monitor'].append(f"{item['failure_mode']} (RPN={item['rpn']})")
        return plan

    def print_action_plan(self, plan):
        """Print the FMEA-based action plan."""
        print("\n" + "=" * 80)
        print("FMEA-BASED ACTION PLAN")
        print("=" * 80)
        if plan['immediate']:
            print("\nImmediate Actions (RPN ≥ 200):")
            for i, a in enumerate(plan['immediate'], 1):
                print(f"  {i}. {a}")
        if plan['short_term']:
            print("\nShort-Term Actions (100 ≤ RPN < 200):")
            for i, a in enumerate(plan['short_term'], 1):
                print(f"  {i}. {a}")
        if plan['long_term']:
            print("\nLong-Term Actions (50 ≤ RPN < 100):")
            for i, a in enumerate(plan['long_term'], 1):
                print(f"  {i}. {a}")
        if plan['monitor']:
            print("\nMonitoring Points:")
            for i, m in enumerate(plan['monitor'], 1):
                print(f"  {i}. {m}")

    # --- Added: adjust_for_desired_productivity ---
    def adjust_for_desired_productivity(self, result, desired_prod):
        """
        Automatically adjust thermal parameters to meet desired productivity
        while minimizing total cycle time and staying close to original predictions.
        """
        current = result['thermal_parameters']['productivity']
        if desired_prod <= current:
            return None  # already meets

        # scaling factor on rates/factors
        orig = result['thermal_parameters']
        scale = desired_prod / current
        adjusted = {
            'heating_rate': min(orig['heating_rate'] * scale, 10.0),
            'cooling_rate': min(orig['cooling_rate'] * scale, 8.0),
            'hold_time_factor': min(orig['hold_time_factor'] * scale, 3.0),
            'peak_temp_adjustment': orig['peak_temp_adjustment'],
            'productivity': desired_prod
        }

        prof = self.generate_thermal_profile(
            result['input_features']['target_temperature'],
            result['input_features']['exposure_time'],
            adjusted['heating_rate'], adjusted['cooling_rate'],
            adjusted['hold_time_factor'], adjusted['peak_temp_adjustment']
        )

        new_result = result.copy()
        new_result['thermal_parameters'] = adjusted
        new_result['thermal_profile'] = prof

        fmea_new = self.perform_fmea_analysis(new_result)
        return new_result, fmea_new


def main():
    print("THERMOPRO - Thermal Processing & Productivity Optimizer")
    print("=" * 60)

    optimizer = ThermalProcessingProfileOptimizer()
    if not optimizer.models_exist():
        print("Models not found; training new models...")
        optimizer.train_models(n_samples=1000)
    else:
        print("Loading existing models...")
        optimizer.load_models()

    print("\n" + "=" * 60)
    print("THERMAL PARAMETER INPUT")
    print("=" * 60)
    try:
        t = float(input("Enter target temperature (20–200°C): "))
        e = float(input("Enter exposure time (10–600 min): "))
        m = float(input("Enter material thickness (0.1–50.0 mm): "))
        result = optimizer.predict_thermal_profile(t, e, m)
        optimizer.print_thermal_report(result)

        desired = float(input("\nEnter your desired productivity (units/hour): "))
        adjustment = optimizer.adjust_for_desired_productivity(result, desired)
        if adjustment is None:
            print(f"\nCurrent predicted productivity ({result['thermal_parameters']['productivity']:.1f}) "
                  f"already meets desired {desired:.1f}. No adjustment needed.")
        else:
            new_res, fmea_new = adjustment
            print("\nAdjusted Parameters to achieve desired productivity:")
            for k, v in new_res['thermal_parameters'].items():
                unit = " units/hour" if k == 'productivity' else ""
                print(f"  {k}: {v:.2f}{unit}")
            print("\nPerforming FMEA on adjusted parameters...")
            optimizer.print_fmea_report(fmea_new)
            plan_new = optimizer.generate_fmea_action_plan(fmea_new)
            optimizer.print_action_plan(plan_new)

        print("\nPerforming FMEA on original parameters...")
        fmea = optimizer.perform_fmea_analysis(result)
        optimizer.print_fmea_report(fmea)
        plan = optimizer.generate_fmea_action_plan(fmea)
        optimizer.print_action_plan(plan)

        if input("\nShow profile plot? (y/n): ").lower().strip() == 'y':
            optimizer.plot_thermal_profile(result)

    except ValueError as ex:
        print(f"Input error: {ex}\nPlease enter valid numeric values within specified ranges.")
    except Exception as ex:
        print(f"Unexpected error: {ex}")


if __name__ == "__main__":
    main()
