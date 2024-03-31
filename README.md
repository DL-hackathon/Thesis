## Thesis: _Deep Learning for Power Quality Disturbances Classification According to Oscillogram Data_

HSE University, Faculty of Computer Science

Educational Programme: Master of Data Science (Master)

Year of Graduation: 2024

In the thesis, oscillograms from a real power grid are utilized to obtain a dataset to train several models based on neural networks (a multilayer perceptron model, a convolution-based model and a gated recurrent unit model). The evaluation focused on the task of multi-class classification of electrical power disturbances. The evaluation clearly demonstrated the superiority of the gated recurrent units over a CatBoost classifier (baseline) and other types of the architectures considered, namely: multilayer perceptron and convolutional neural network.

Source code files for Thesis:
- dataset_helper_functions.py  - functions to obtain training, validation and test datasets from raw data (oscillograms);
- training_helper_functions.py - functions to train several neural networks of different architectures.
- thesis.ipynb                 - Jupyter notebook of the research. 
