# Wakulima FarmTech Intelligence

> An end-to-end ML/DL platform delivering precision agriculture
> intelligence to smallholder farmers across Africa — covering
> crop selection, soil fertility, and plant disease diagnosis.

![Python](https://img.shields.io/badge/Python-3.9-blue?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-deployed-green?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DL-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## Problem Statement

Smallholder farmers in sub-Saharan Africa face three compounding
challenges that suppress yields and income:

- **Crop selection uncertainty** — planting unsuitable crops for
  local soil and climate conditions leads to failed harvests
- **Fertilizer mismanagement** — nutrient over- or under-application
  degrades soil health and inflates input costs by 20–40%
- **Late disease diagnosis** — undetected crop disease causes
  an estimated 20–40% annual yield loss in East Africa

These are not knowledge problems. They are data access problems.
Wakulima FarmTech turns soil science and computer vision into
decisions any farmer can act on.

---

## System Architecture

```
User Input (Soil / Image)
        │
        ▼
   Flask Web App (app.py)
        │
   ┌────┴──────────────────────┐
   │                           │
   ▼                           ▼
Crop & Fertilizer         Disease Detection
Recommendation Engine     (CNN — TensorFlow/Keras)
(scikit-learn / Random     VGG16 Transfer Learning
 Forest + weather API)     14 crop classes
        │                           │
        ▼                           ▼
   Soil nutrient report    Disease label + treatment
   + crop recommendation   plan served to user
```

---

## Core Modules

### 1. Crop Recommendation Engine
Predicts the optimal crop based on real-time agronomy data.

**Inputs:** N, P, K ratios · pH · temperature · humidity
(live-fetched via OpenWeatherMap API)

**Model:** Random Forest classifier trained on regional soil
datasets · >95% accuracy on validation set

**Output:** Top crop recommendation with agronomic rationale

---

### 2. Fertilizer Advisory System
Identifies soil nutrient gaps and prescribes corrective action.

**Inputs:** Soil N-P-K levels · target crop type

**Logic:** Rule-based deficiency/excess detection layered on
ML classification — provides specific fertilizer product
recommendations, not just nutrient labels

**Output:** Deficiency/excess report + fertilizer prescription

---

### 3. Plant Disease Detection (CNN)
Classifies leaf images to identify disease and recommend treatment.which we are in development 

**Model:** VGG16 fine-tuned on PlantVillage dataset
**Classes:** 14 crops · 38 disease/healthy categories
**Status:** Beta — expanding training data for Kenyan varieties

Supported crops: Apple · Blueberry · Cherry · Corn · Grape ·
Orange · Peach · Pepper · Potato · Raspberry · Soybean ·
Squash · Strawberry · Tomato

---

## Tech Stack

| Layer         | Technology                                      |
|---------------|-------------------------------------------------|
| ML Models     | scikit-learn, Random Forest, XGBoost            |
| Deep Learning | TensorFlow, Keras, VGG16 (transfer learning)    |
| Backend       | Python 3.9, Flask, REST API                     |
| Deployment    | Procfile (Heroku-ready), Azure-compatible       |
| Data          | Pandas, NumPy, OpenWeatherMap API               |
| Frontend      | HTML/CSS/JS, Jinja2 templates                   |
| Versioning    | Git, GitHub                                     |

---

## Project Structure

```
WAKULIMA_FARMTECH_INTELLIGENCE/
│
├── app.py                  # Main Flask application
├── config.py               # Environment config
├── requirements.txt        # Dependencies
├── Procfile                # Deployment config
│
├── models/                 # Serialized ML & DL models
├── notebooks/              # EDA, training notebooks
├── src/                    # Core prediction logic
├── data/                   # Training datasets
├── static/                 # CSS, JS, images
├── templates/              # HTML templates (Jinja2)
├── utils/                  # Helper functions
└── tests/                  # Unit tests
```

---

## Quickstart

```bash
# 1. Clone the repository
git clone https://github.com/samuelMG1/WAKULIMA_FARMTECH_INTELLIGENCE.git
cd WAKULIMA_FARMTECH_INTELLIGENCE

# 2. Create and activate conda environment
conda create -n wakulima python=3.9
conda activate wakulima

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your OpenWeatherMap API key
export WEATHER_API_KEY=your_api_key_here

# 5. Run the application
python app.py
# Visit http://localhost:5000
```

---

## Impact & Relevance

This system addresses the exact intersection where data science
meets agricultural development in Africa:

- Aligns with Kenya's **Big 4 Agenda** on food security
- Built for **low-data environments** — works with minimal sensor
  input, no smartphone camera required for soil modules
- Designed with **smallholder farmers as primary users** —
  simple UI, actionable outputs, no agronomist needed
- Extensible to **IoT sensor ingestion** for real-time soil
  monitoring pipelines

---
Designed and built of the full system: data pipelines, ML model
training, CNN architecture, Flask deployment, and API integration.

> An end-to-end ML/DL platform delivering precision agriculture
> intelligence to smallholder farmers across Africa — covering
> crop selection, soil fertility, and plant disease diagnosis.

![Python](https://img.shields.io/badge/Python-3.9-blue?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-deployed-green?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DL-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## Problem Statement

Smallholder farmers in sub-Saharan Africa face three compounding
challenges that suppress yields and income:

- **Crop selection uncertainty** — planting unsuitable crops for
  local soil and climate conditions leads to failed harvests
- **Fertilizer mismanagement** — nutrient over- or under-application
  degrades soil health and inflates input costs by 20–40%
- **Late disease diagnosis** — undetected crop disease causes
  an estimated 20–40% annual yield loss in East Africa

These are not knowledge problems. They are data access problems.
Wakulima FarmTech turns soil science and computer vision into
decisions any farmer can act on.

---

## System Architecture

```
User Input (Soil / Image)
        │
        ▼
   Flask Web App (app.py)
        │
   ┌────┴──────────────────────┐
   │                           │
   ▼                           ▼
Crop & Fertilizer         Disease Detection
Recommendation Engine     (CNN — TensorFlow/Keras)
(scikit-learn / Random     VGG16 Transfer Learning
 Forest + weather API)     14 crop classes
        │                           │
        ▼                           ▼
   Soil nutrient report    Disease label + treatment
   + crop recommendation   plan served to user
```

---

## Core Modules

### 1. Crop Recommendation Engine
Predicts the optimal crop based on real-time agronomy data.

**Inputs:** N, P, K ratios · pH · temperature · humidity
(live-fetched via OpenWeatherMap API)

**Model:** Random Forest classifier trained on regional soil
datasets · >95% accuracy on validation set

**Output:** Top crop recommendation with agronomic rationale

---

### 2. Fertilizer Advisory System
Identifies soil nutrient gaps and prescribes corrective action.

**Inputs:** Soil N-P-K levels · target crop type

**Logic:** Rule-based deficiency/excess detection layered on
ML classification — provides specific fertilizer product
recommendations, not just nutrient labels

**Output:** Deficiency/excess report + fertilizer prescription

---

### 3. Plant Disease Detection (CNN)
Classifies leaf images to identify disease and recommend treatment.

**Model:** VGG16 fine-tuned on PlantVillage dataset
**Classes:** 14 crops · 38 disease/healthy categories
**Status:** Beta — expanding training data for Kenyan varieties

Supported crops: Apple · Blueberry · Cherry · Corn · Grape ·
Orange · Peach · Pepper · Potato · Raspberry · Soybean ·
Squash · Strawberry · Tomato

---

## Tech Stack

| Layer         | Technology                                      |
|---------------|-------------------------------------------------|
| ML Models     | scikit-learn, Random Forest, XGBoost            |
| Deep Learning | TensorFlow, Keras, VGG16 (transfer learning)    |
| Backend       | Python 3.9, Flask, REST API                     |
| Deployment    | Procfile (Heroku-ready), Azure-compatible       |
| Data          | Pandas, NumPy, OpenWeatherMap API               |
| Frontend      | HTML/CSS/JS, Jinja2 templates                   |
| Versioning    | Git, GitHub                                     |

---

## Project Structure

```
WAKULIMA_FARMTECH_INTELLIGENCE/
│
├── app.py                  # Main Flask application
├── config.py               # Environment config
├── requirements.txt        # Dependencies
├── Procfile                # Deployment config
│
├── models/                 # Serialized ML & DL models
├── notebooks/              # EDA, training notebooks
├── src/                    # Core prediction logic
├── data/                   # Training datasets
├── static/                 # CSS, JS, images
├── templates/              # HTML templates (Jinja2)
├── utils/                  # Helper functions
└── tests/                  # Unit tests
```

---

## Quickstart

```bash
# 1. Clone the repository
git clone https://github.com/samuelMG1/WAKULIMA_FARMTECH_INTELLIGENCE.git
cd WAKULIMA_FARMTECH_INTELLIGENCE

# 2. Create and activate conda environment
conda create -n wakulima python=3.9
conda activate wakulima

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your OpenWeatherMap API key
export WEATHER_API_KEY=your_api_key_here

# 5. Run the application
python app.py
# Visit http://localhost:5000
```

---

## Impact & Relevance

This system addresses the exact intersection where data science
meets agricultural development in Africa:

- Aligns with Kenya's **Big 4 Agenda** on food security
- Built for **low-data environments** — works with minimal sensor
  input, no smartphone camera required for soil modules
- Designed with **smallholder farmers as primary users** —
  simple UI, actionable outputs, no agronomist needed
- Extensible to **IoT sensor ingestion** for real-time soil
  monitoring pipelines

---

## Roadmap

- [ ] Retrain disease model on East African crop varieties
- [ ] Add Swahili language UI option
- [ ] Integrate IoT soil sensor data ingestion (MQTT pipeline)
- [ ] REST API endpoints for third-party agri-app integration
- [ ] Mobile-first PWA version for low-bandwidth environments

---

## License

MIT License — open for collaboration and adaptation for
agricultural development use cases.

---

## Roadmap

- [ ] Retrain disease model on East African crop varieties
- [ ] Add Swahili language UI option
- [ ] Integrate IoT soil sensor data ingestion (MQTT pipeline)
- [ ] REST API endpoints for third-party agri-app integration
- [ ] Mobile-first PWA version for low-bandwidth environments

---

## License

MIT License — open for collaboration and adaptation for
agricultural development use cases.