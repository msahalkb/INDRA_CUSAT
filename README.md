# GlucoTwin: A Digital Twin for 2-Hour Glucose Spike Prediction in Type 2 Diabetes

> Happiest Health "Digital Twin Challenge" - Phase 1 submission

## 1. Team Details (Team INDRA)
| Name | Role | College / Incubator | Email |
|---|---|---|---|
| <Leader name> (team leader) | Team Leader | Cochin University of Science and Technology (CUSAT), Kochi | msahalkb@gmail.com |

## 2. Project Title
GlucoTwin: A Digital Twin for 2-Hour Glucose Spike Prediction in Type 2 Diabetes

## 3. Problem Statement and Healthcare Use Case
Type 2 diabetes affects over 100 million people in India. Post-meal glucose spikes drive complications, but patients and doctors only see them after they happen. This prototype is a virtual patient that predicts, from the last few hours of data plus the patient's clinical record, whether glucose will rise above 180 mg/dL within the next 2 hours.

## 4. Technical Stack and Model Details
- Python, pandas, scikit-learn, LightGBM, PyTorch, Streamlit
- Static stream: EHR-style clinical features (demographics, HbA1c, C-peptide, eGFR, medications, comorbidities)
- Dynamic stream: CGM every 15 min, logged meals and insulin, simulated heart rate and steps
- Models: gradient boosting baseline; GRU with static-feature fusion
- Evaluation: patient-grouped cross-validation, AUROC, precision/recall, lead time, ablation (dynamic only / static only / fused)

## 5. Data Sources and Honesty Statement
- Real: Shanghai T2DM dataset (100 patients, CGM, meals, insulin, clinical summary).
- Simulated: heart rate and step count (generated, labelled as such).
- Limitations: East Asian cohort, 3-14 days per patient, simulated wearables, not clinically validated.

## 6. Video (20+ minutes)
<link to unlisted YouTube video or file in repo>

## 7. Open-Source License
MIT. See [LICENSE](LICENSE). Dataset license and citation are listed in `data/README.md`.

## 8. Architecture Diagram
[docs/architecture.pdf](docs/architecture.pdf)

## 9. Presentation
[docs/presentation.pdf](docs/presentation.pdf)

## Repository Structure
```
data/        data instructions (raw data not committed)
src/         data pipeline, wearable simulator, models
notebooks/   exploration
app/         Streamlit doctor dashboard
docs/        architecture diagram, slides
models/      saved models
results/     metrics and plots
```

## How to Run
```
pip install -r requirements.txt
# 1. place data as described in data/README.md
# 2. python src/build_dataset.py
# 3. python src/train.py
# 4. streamlit run app/dashboard.py
```
