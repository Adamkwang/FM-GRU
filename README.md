# FM-GRU
A recurrent neural network water quality prediction method based on a sequence-to-sequence framework.More details(including parameters settings) refer to the original paper.
## Paper
FM-GRU: A Time Series Prediction Method for Water Quality Based on Seq2Seq Framework
## Dataset
The dataset we use is a variety of national water quality reference index data monitored by a monitoring station in Lianjiang, Shantou City, China from January 1, 2019 to June 30, 2020. Due to the different characteristics of different water quality indicators, the monitoring stations have different monitoring frequencies for different
indicators. It is obvious that using data with different monitoring frequencies will affect
the performance of the model. Therefore, we selected five water quality indicators with
the same monitoring frequency (all one hour) from this dataset: water temperature, pH,
conductivity, turbidity, and dissolved oxygen. The dissolved oxygen was used as the
target for prediction, and the remaining four indicators were used as covariates.
## Getting Started
### Prerequisites
* python 3.5+
* python libraries
> numpy
> pandas
> pytorch
> scipy
> sklearn

## Run
'python main.py'

## Shantou University, Shantou, China
