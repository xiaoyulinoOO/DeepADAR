# DeepADAR
## A general and accurate deep learning-based method for easily identifying Pogostemon cablin categories in hyperspectral data.
The hyperspectral data of Pogostemon cablin contains rich information about Pogostemon cablin but is difficult to analyze. In this study, a method called DeepADAR was proposed to address this issue. Essentially, it is a deep learning model that combines Convolutional Neural Networks (CNN), Long Short-Term Memory Networks (LSTM), and Attention Mechanisms to compare the spectral data of Pogostemon cablin from different origins. DeepADAR achieved promising results in the analysis of hyperspectral data for three categories of Pogostemon cablin. In summary, it is an accurate, versatile, and ready-to-use method that can be applied to origin identification and adulteration detection of Pogostemon cablin in various scenarios.
# Depends
``pip install -r requirements.txt``
# Usage
1.Training your model

Run the file 'training.py'. Since the data exceeded the limit, we have uploaded some example data for training，download at releases.

2.Predict mixture spectra

Run the file 'testing.py'. More example data for testing can be download at releases.
