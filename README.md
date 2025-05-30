# Landslide Susceptibility Mapping Using Deep Learning

## Overview

This repository hosts the implementation of a Deep Learning-based approach to map landslide susceptibility in Himachal Pradesh, India. By leveraging advanced geospatial data and deep learning models, the project aims to provide actionable insights for risk management and disaster mitigation.

## Repository

GitHub Repository: https://github.com/SharmaS1402/Landslide_Susceptibility_Mapping_Deep_Learning

## Project Highlights

1. High-Resolution Landslide Susceptibility Maps: Built using geomorphological and environmental factors.

2. Innovative Deep Learning Models: Applied to large-scale geospatial data for accurate predictions.

3. Risk Assessment and Planning: Results can inform infrastructure development and disaster management policies.

## Objective

To develop a high-resolution landslide susceptibility map for Himachal Pradesh utilizing deep learning models to:

1. Evaluate and predict landslide risks for infrastructure and community safety.

2. Provide scientific insights to enhance disaster risk reduction strategies.

## Study Area

Location: Himachal Pradesh, India.

Importance: This region experiences frequent landslides due to steep slopes, rugged terrain, and environmental challenges.

## Methodology

### Data Acquisition

#### Sources:

1. High-resolution Digital Elevation Model (DEM).

2. Historical landslide data.

3. Environmental data such as precipitation, vegetation, and land use/land cover.

#### Key Factors:

Geomorphological: Altitude, slope, aspect, ruggedness index.

Environmental: Vegetation cover, precipitation, land use.

### Data Preparation

Point Sampling: Extracted feature values using QGIS Point Sampling Tool.

Data Encoding: Organized data into a tabular format with binary labels (1: landslide, 0: no landslide).

Train-Test Split: 80% training and 20% testing.

### Deep Learning Model

Architecture:

Input Layer: 7 neurons corresponding to the input features.

Hidden Layers: Three dense layers with 30, 10, and 870 neurons using ReLU activation.

Regularization: Batch normalization and dropout (rate: 0.5) to enhance stability and reduce overfitting.

Output Layer: Single neuron with a sigmoid activation function for binary classification.

Loss Function: Binary Cross-Entropy.

Optimizer: Adam.

Performance: Achieved 87.84% accuracy on the test set.

## Results

1. Predicted landslide susceptibility values for 10,000 geospatial points.

2. Generated high-resolution susceptibility maps for Himachal Pradesh.

3. Demonstrated superior performance compared to traditional methods.

## Key Contributions

1. Advanced Model Application: Successfully applied a Deep Neural Network (DNN) to geospatial landslide susceptibility mapping.

2. Scalability and Precision: Addressed challenges in handling high-dimensional datasets from various sources.

3. Actionable Outputs: Created maps to assist policymakers, urban planners, and emergency services.

## Mitigation Strategies

1. Enhanced monitoring systems for high-risk zones.

2. Awareness programs to educate communities about landslide risks.

3. Incorporation of susceptibility maps into regional planning and development strategies.

## References

QGIS Development Team. "QGIS Geographic Information System." Open Source Geospatial Foundation Project.

U.S. Geological Survey. "Earth Explorer."

National Remote Sensing Centre. "Bhuvan: An Indian Geo-Platform."

Additional academic studies cited in the project presentation.

### Contributors
SharmaS1402
rizul22

## License

This project is licensed under the MIT License. See the LICENSE file for details.

For more details visit the /Main/reports directory which includes project report and a powerpoint presentation.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project
├── data
│   ├── external       <- Data from third party sources
│   ├── interim        <- Intermediate data that has been transformed
│   ├── processed      <- The final, canonical data sets for modeling
│   └── raw            <- The original, immutable data dump
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── plots          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── src                         <- Source code for this project
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    │    
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    ├── plots.py                <- Code to create visualizations 
    │
    └── services                <- Service classes to connect with external platforms, tools, or APIs
        └── __init__.py 
```

--------
