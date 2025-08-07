import matplotlib.pyplot as plt
import numpy as np

# Input data
data_str = """
Results (MMBench_DEV_EN)
--------------------------------------  -------------------- This is the result of llavanext-scaled-0.5b 
split                                   dev
Overall                                 0.5420962199312714
AR                                      0.6331658291457286
CP                                      0.6554054054054054
FP-C                                    0.44755244755244755
FP-S                                    0.5426621160409556
LR                                      0.2966101694915254
RR                                      0.4608695652173913
action_recognition                      0.8703703703703703
attribute_comparison                    0.22727272727272727
attribute_recognition                   0.6486486486486487
celebrity_recognition                   0.6767676767676768
function_reasoning                      0.7088607594936709
future_prediction                       0.4
identity_reasoning                      0.8888888888888888
image_emotion                           0.72
image_quality                           0.018867924528301886
image_scene                             0.9326923076923077
image_style                             0.6226415094339622
image_topic                             0.75
nature_relation                         0.3958333333333333
object_localization                     0.2222222222222222
ocr                                     0.6666666666666666
physical_property_reasoning             0.4
physical_relation                       0.20833333333333334
social_relation                         0.6744186046511628
spatial_relationship                    0.15555555555555556
structuralized_imagetext_understanding  0.24358974358974358
--------------------------------------  --------------------
--------------------------------------  -------------------- This is the result of llava-onevision-qwen2-0.5b-si
Overall                                 0.5506872852233677
AR                                      0.5979899497487438
CP                                      0.6959459459459459
FP-C                                    0.3776223776223776
FP-S                                    0.590443686006826
LR                                      0.23728813559322035
RR                                      0.5304347826086957
action_recognition                      0.8703703703703703
attribute_comparison                    0.0
attribute_recognition                   0.7432432432432432
celebrity_recognition                   0.7171717171717171
function_reasoning                      0.7468354430379747
future_prediction                       0.375
identity_reasoning                      0.9333333333333333
image_emotion                           0.82
image_quality                           0.05660377358490566
image_scene                             0.9326923076923077
image_style                             0.7358490566037735
image_topic                             0.7222222222222222
nature_relation                         0.3541666666666667
object_localization                     0.24691358024691357
ocr                                     0.6923076923076923
physical_property_reasoning             0.24
physical_relation                       0.4166666666666667
social_relation                         0.7906976744186046
spatial_relationship                    0.15555555555555556
structuralized_imagetext_understanding  0.16666666666666666
--------------------------------------  --------------------
"""

# Parse data
lines = data_str.strip().split('\n')
results = {}
current_model = None

for line in lines:
    if "This is the result of" in line:
        current_model = line.split("This is the result of")[-1].strip()
        results[current_model] = {}
    elif "---" not in line and line.strip():
        parts = line.rsplit(maxsplit=1)
        if len(parts) == 2:
            key = parts[0].strip()
            try:
                value = round(float(parts[1].strip()), 3)
                results[current_model][key] = value
            except ValueError:
                continue

# Extract model names
model_names = list(results.keys())
model1, model2 = model_names[0], model_names[1]

# Category definitions
general_categories = ['AR', 'CP', 'FP-C', 'FP-S', 'LR', 'RR']
specific_categories = [
    'action_recognition', 'attribute_comparison', 'attribute_recognition',
    'celebrity_recognition', 'function_reasoning', 'future_prediction',
    'identity_reasoning', 'image_emotion', 'image_quality', 'image_scene',
    'image_style', 'image_topic', 'nature_relation', 'object_localization',
    'ocr', 'physical_property_reasoning', 'physical_relation',
    'social_relation', 'spatial_relationship', 
    'structuralized_imagetext_understanding'
]

# Prepare data for plotting
def get_values(model, categories):
    return [results[model].get(cat, 0) for cat in categories]

# Create general tests comparison
plt.figure(figsize=(12, 6))
x = np.arange(len(general_categories))
width = 0.35

model1_vals = get_values(model1, general_categories)
model2_vals = get_values(model2, general_categories)

plt.bar(x - width/2, model1_vals, width, label=model1)
plt.bar(x + width/2, model2_vals, width, label=model2)

plt.ylabel('Scores')
plt.title('General Tests Comparison')
plt.xticks(x, general_categories)
plt.legend()
plt.ylim(0, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
for i, (v1, v2) in enumerate(zip(model1_vals, model2_vals)):
    plt.text(i - width/2, v1 + 0.02, f'{v1:.3f}', ha='center')
    plt.text(i + width/2, v2 + 0.02, f'{v2:.3f}', ha='center')

plt.tight_layout()
plt.savefig('general_tests_comparison.png')
plt.close()

# Create specific tests comparison
plt.figure(figsize=(16, 8))
x = np.arange(len(specific_categories))
width = 0.35

model1_vals = get_values(model1, specific_categories)
model2_vals = get_values(model2, specific_categories)

plt.bar(x - width/2, model1_vals, width, label=model1)
plt.bar(x + width/2, model2_vals, width, label=model2)

plt.ylabel('Scores')
plt.title('Specific Tests Comparison')
plt.xticks(x, [cat.replace('_', ' ').title() for cat in specific_categories], rotation=90)
plt.legend()
plt.ylim(0, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels above bars
for i, (v1, v2) in enumerate(zip(model1_vals, model2_vals)):
    plt.text(i - width/2, v1 + 0.02, f'{v1:.3f}', ha='center', fontsize=8)
    plt.text(i + width/2, v2 + 0.02, f'{v2:.3f}', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('specific_tests_comparison.png', bbox_inches='tight')
plt.close()

print("Bar charts saved as 'general_tests_comparison.png' and 'specific_tests_comparison.png'")