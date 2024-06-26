import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import importlib
from sklearn.metrics import f1_score, precision_score, recall_score

data_df=pd.read_csv("fuzzy/data/diabetes.csv")
clean_data_df = data_df[
    (data_df['Glucose'] != 0) &
    (data_df['BloodPressure'] != 0) &
    (data_df['SkinThickness'] != 0) &
    (data_df['Insulin'] != 0) &
    (data_df['BMI'] != 0)
]

def buildRules(
    glucose: ctrl.Antecedent,
    age: ctrl.Antecedent,
    bmi: ctrl.Antecedent,
    prediction: ctrl.Antecedent,
):
    return [
        # glucose, age, no
        ctrl.Rule(glucose['very low'] &
                  (age['young'] | age['middle age'] | age['old']),
                  prediction['no']),
        ctrl.Rule(glucose['low'] &
                  (age['young'] | age['middle age']),
                  prediction['no']),
        ctrl.Rule(glucose['medium'] &
                  (age['young'] | age['middle age']),
                  prediction['no']),
        ctrl.Rule(glucose['high'] &
                  age['young'],
                  prediction['no']),

        # glucose, age, yes
        ctrl.Rule(glucose['high'] &
                  age['old'],
                  prediction['yes']),
        ctrl.Rule(glucose['very high'] &
                  (age['young'] | age['middle age'] | age['very old']),
                  prediction['yes']),

        # glucose, bmi, no
        ctrl.Rule(glucose['very low'] &
                  (bmi['normal weight'] | bmi['overweight'] | bmi['obesity 2']),
                  prediction['no']),
        ctrl.Rule(glucose['low'] &
                  (bmi['underweight'] | bmi['normal weight'] | bmi['overweight'] | bmi['obesity 1'] | bmi['obesity 2']),
                  prediction['no']),
        ctrl.Rule(glucose['medium'] &
                  (bmi['normal weight'] | bmi['overweight'] | bmi['obesity 2']),
                  prediction['no']),
        ctrl.Rule(glucose['high'] &
                  (bmi['normal weight']),
                  prediction['no']),

        # glucose, bmi, yes
        ctrl.Rule(glucose['very high'] &
                  (bmi['normal weight'] | bmi['overweight'] | bmi['obesity 1'] | bmi['obesity 2']),
                  prediction['yes']),
    ]


def build_fuzzy_inference_system(defuzzify_method: str = 'centroid'):
    glucose = ctrl.Antecedent(np.arange(56, 198+1, 0.01), 'glucose')
    age = ctrl.Antecedent(np.arange(21, 81+1, 0.01), 'age')
    bmi = ctrl.Antecedent(np.arange(18.2, 67.1+0.1, 0.01), 'bmi')
    prediction = ctrl.Consequent(np.arange(0, 1+0.05, 0.05), 'prediction',
                                 defuzzify_method=defuzzify_method)

    # * Membership Functions
    # glucose
    glucose['very low'] = fuzz.trapmf(glucose.universe, [0, 53, 61.5, 97.5])
    glucose['low'] = fuzz.trimf(glucose.universe, [53, 83.75, 125])
    glucose['medium'] = fuzz.trimf(glucose.universe, [70, 111.25, 162.5])
    glucose['high'] = fuzz.trimf(glucose.universe, [97.5, 143.75, 200])
    glucose['very high'] = fuzz.trapmf(glucose.universe, [125, 181.25, 200, 300])

    # age
    age['young'] = fuzz.trapmf(age.universe, [0, 19, 21, 32.5])
    age['middle age'] = fuzz.trapmf(age.universe, [20, 27.25, 37.75, 47])
    age['old'] = fuzz.trapmf(age.universe, [32.5, 45, 49, 66.5])
    age['very old'] = fuzz.trapmf(age.universe, [47, 58.75, 100, 100])

    # bmi
    bmi['underweight'] = fuzz.trapmf(bmi.universe, [0, 0, 18.25, 21.75])
    bmi['normal weight'] = fuzz.trimf(bmi.universe, [18.25, 21.75, 27.5])
    bmi['overweight'] = fuzz.trimf(bmi.universe, [21.75, 27.5, 32.5])
    bmi['obesity 1'] = fuzz.trimf(bmi.universe, [27.5, 32.5, 37.5])
    bmi['obesity 2'] = fuzz.trimf(bmi.universe, [32.5, 37.5, 42.5])
    bmi['obesity 3'] = fuzz.trapmf(bmi.universe, [37.5, 42.5, 80, 80])

    # output
    prediction['no'] = fuzz.trimf(prediction.universe, [0, 0, 0.8])
    prediction['yes'] = fuzz.trimf(prediction.universe, [0.2, 1, 1])

    # * Rules
    rules = buildRules(
        glucose=glucose,
        age=age,
        bmi=bmi,
        prediction=prediction,
    )

    # * Mamdani control system
    prediction_ctrl = ctrl.ControlSystem(rules)
    prediction_inference = ctrl.ControlSystemSimulation(prediction_ctrl)

    return prediction_inference, prediction


if __name__ == '__main__':
    prediction_inference, prediction = build_fuzzy_inference_system()
    classification_threshold = 0.5

    prediction_inference.input['glucose'] = clean_data_df['Glucose'].to_numpy()
    prediction_inference.input['age'] = clean_data_df['Age'].to_numpy()
    prediction_inference.input['bmi'] = clean_data_df['BMI'].to_numpy()

    prediction_inference.compute()
    result_np = prediction_inference.output['prediction']

    output_np = np.where(result_np > 0.50, 1, 0)
    gt_np = clean_data_df['Outcome'].to_numpy()
    total_sample = gt_np.shape[0]
    accurate_sample = (output_np == gt_np).sum()
    accuracy = (accurate_sample / total_sample) * 100
    print(f"Total samples: {total_sample}")
    print(f"Correctly classified samples: {accurate_sample}")
    print(f"Accuracy: {accuracy:.2f}%")

    precision = precision_score(gt_np, output_np)
    recall = recall_score(gt_np, output_np)
    f1 = f1_score(gt_np, output_np)

    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')