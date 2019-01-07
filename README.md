# HTM_Streaming_Anomaly

Using Hierarchical Temporal Memory (HTM) for streaming anomaly detection.

This program utilizes Numenta Platform for Intelligent Computing's (NuPIC's) implementation of HTM on streaming datasets from the Numenta Anomaly Benchmark (NAB).

The two datasets I used can be found at:
https://github.com/numenta/NAB/blob/master/data/realKnownCause/machine_temperature_system_failure.csv

https://github.com/numenta/NAB/blob/master/data/realTweets/Twitter_volume_GOOG.csv

Usage (cd into the anomaly folder):
python run.py --dataset 0

python run.py --dataset 1

The --dataset 0 option will run the algorithm on the machine temperature dataset while --dataset 1 option will run the algorithm on the Twitter Google traffic dataset.

----------------------------------------------------------------------------------------------------------------------------

Extra details:

Swarm

A swarm algorithm was run to find ideal hyperparameters for the HTM model. The swarm algorithm assesses the input data to determine hyperparameters such as the number of mini-columns to use, the number of active columns per inhbition area, the increment and decrement amounts for the permanence values of each mini-column and cell, etc. Once the ideal hyperparameters are found the algorithm is run on the datasets to predict a single time step into the future at each time step. By predicting the next time step, HTM is able to determine if the next piece of data is what it's expecting. If the data is not expected it evaluates a number of time steps prior to determine if the current time step is an anomaly or not.

Output Plots

The blue streams indicate the original data over time while the orange streams indicate the next step prediction that HTM is conducting to determine what data is expected. From these predictions the HTM algorithm determines an anomaly score, indicated by purple lines, and an anomaly likelihood, indicated by red lines. The anomaly score is calculated by determining the overlap between the predicted mini-columns in the TM and the active mini-columns seen from the current input data. The values for the anomaly score are set between 0 and 1. If the anomaly score is close to 0 then the prediction made by HTM was similar or equivalent to the current data, while if the anomaly score is close to 1 then the prediction made by HTM was shown to differentiate from the pattern predicted and therefore an anomaly may be occurring. The anomaly likelihood estimates a probability distribution relative to anomaly scores from the previous data points. This allows for an anomaly to be detected more readily in noisy datasets as there are high anomaly scores in certain datasets.

Requires:
- NAB:   https://github.com/numenta/NAB
- NuPIC: https://github.com/numenta/nupic
- Python 2.x (I used 2.7.15rc1)
