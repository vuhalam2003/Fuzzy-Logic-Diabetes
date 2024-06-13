import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from xgboost import XGBClassifier

# Define the functions as provided
def draw_y_lines(boundaries: list, ax: Axes):
    for boundary in boundaries:
        ax.axhline(y=boundary, linestyle='--', color='0.6')

def draw_x_lines(boundaries: list, ax: Axes):
    for boundary in boundaries:
        ax.axvline(x=boundary, linestyle='--', color='0.6')

def plot_area_matrix(
    clean_data_df: pd.DataFrame,
    x_feature: str,
    y_feature: str,
    count_feature: str,
    x_boundaries: list,
    y_boundaries: list,
    ax: Axes,
):
    draw_x_lines(boundaries=x_boundaries, ax=ax)
    draw_y_lines(boundaries=y_boundaries, ax=ax)
    sns.scatterplot(data=clean_data_df,
                    x=x_feature,
                    y=y_feature,
                    ax=ax,
                    hue=count_feature,
                    linewidth=1,
                    edgecolor='black',
                    alpha=0.7,
                    style=count_feature)

    for i in range(len(x_boundaries) - 1):
        x_left_bound = x_boundaries[i]
        x_right_bound = x_boundaries[i+1]
        for j in range(len(y_boundaries) - 1):
            y_left_bound = y_boundaries[j]
            y_right_bound = y_boundaries[j+1]
            # filtering & counting
            sub_data_df = clean_data_df[(clean_data_df[y_feature].between(y_left_bound, y_right_bound)) &
                                        (clean_data_df[x_feature].between(x_left_bound, x_right_bound))]
            total_count = sub_data_df.shape[0]
            if total_count < 1:
                continue
            count_map = sub_data_df[count_feature].value_counts()

            if total_count > 0:
                no_count = count_map[0] if 0 in count_map.index else 0
                yes_count = count_map[1] if 1 in count_map.index else 0

                o_y = y_left_bound
                o_x = x_left_bound
                rect_height = y_right_bound - o_y
                rect_width = x_right_bound - o_x

                yes_ratio = yes_count / total_count

                area_color = list(mpl.colormaps['coolwarm'](yes_ratio))
                area_color[3] = 0.8

                rect = Rectangle(
                    (o_x, o_y), rect_width, rect_height, facecolor=area_color
                )
                # label_str = f"{yes_ratio:.2f}\n0: {no_count}\n1: {yes_count}\nTotal: {total_count}"
                label_str = f"{yes_ratio:.2f}\nTotal: {total_count}"
                ax.add_patch(rect)
                ax.text(
                    x=o_x + rect_width / 2,
                    y=o_y + rect_height / 2,
                    s=label_str,
                    color='white',
                    fontsize=10,
                    ha='center',
                    va='center',
                    weight='bold',
                    path_effects=[path_effects.withStroke(linewidth=2, foreground="black")]
                )

# Load the data
data_df = pd.read_csv("data/diabetes.csv")

# Clean the data
clean_data_df = data_df[
    (data_df['Glucose'] != 0) &
    (data_df['BloodPressure'] != 0) &
    (data_df['SkinThickness'] != 0) &
    (data_df['Insulin'] != 0) &
    (data_df['BMI'] != 0)
]

# Plot the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(clean_data_df.corr(), annot=True, cmap="Reds")
plt.title("Correlation Matrix", fontsize=25)
plt.show()

# Define the features and target
X_df = clean_data_df.drop(['Outcome'], axis=1)
Y_df = clean_data_df['Outcome']

# Define and fit the model
model = XGBClassifier()
model.fit(X_df, Y_df)

# Get feature importance
feature_importance = model.feature_importances_

# Normalize the feature importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

# Plot the feature importance
plt.figure(figsize=(10, 8))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, clean_data_df.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Feature Importance')
plt.show()

filter_clean_data_df = clean_data_df[['Glucose', 'Age', 'BMI', 'Outcome']]

