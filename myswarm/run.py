#!/usr/bin/env python

# general
import csv
import datetime
import nupic_output

# model parameters
from machine_model_params import MODEL_PARAMS
machine_model_params = MODEL_PARAMS
from twitter_model_params import MODEL_PARAMS
twitter_model_params = MODEL_PARAMS

# model
from nupic.frameworks.opf.model_factory import ModelFactory

# plotting
from nupic.data.inference_shifter import InferenceShifter

# Global variable for date format
DATE_FORMAT = "%m/%d/%Y %H:%M"
# ex) 2/3/2001 21:45 

def createModel(model_par):
    """
    Creates the HTM model
    
    :param model_params : parameters for model
    """
    model = ModelFactory.create(model_par)
    model.enableInference({
        "predictedField": "value"
        })
    return model

def runModel(model, csv_path, outputCSVFile, outputPlotFile):
    """
    Runs HTM model with input data

    :param model    : input HTM model
    :param csv_path : path to csv dataset file
    :param outputCSVFile : output csv file
    :param outputPlotFile: output plot file
    """

    # get input csv file and read it
    inputFilePath = csv_path
    inputFile = open(inputFilePath, "rb")
    csvReader = csv.reader(inputFile)

    # skip first 3 header rows
    csvReader.next()
    csvReader.next()
    csvReader.next()

    # plot prediction
    shifter = InferenceShifter()

    # loop through data
    counter = 0
    for row in csvReader:
        counter += 1
        # print after every 100 iterations
        if (counter % 100 == 0):
            print("Read %i lines..." % counter)
        timestamp = datetime.datetime.strptime(row[0], DATE_FORMAT)
        value = float(row[1])
        result = model.run({
            "timestamp":timestamp,
            "value":value
            })
        
        # if prediction instead of anomaly detection; 1 prediction step
        plot_result = shifter.shift(result)
        plot_prediction = plot_result.inferences["multiStepBestPredictions"][1]
        outputPlotFile.write([timestamp], [value], [plot_prediction])

        prediction = result.inferences["multiStepBestPredictions"][1]
        outputCSVFile.write([timestamp], [value], [prediction])

    # close all files after usage
    inputFile.close()
    outputCSVFile.close()
    outputPlotFile.close()

    return result


def runDataset(dataset):
    """
    Runs through the dataset given for anomaly detection

    """

    # set model parameters, csv path, and output csv/plot
    if(dataset == 0):
        model_par = machine_model_params
        csv_path = "./data/machine_temperature_system_failure.csv"
        outputCSVFile = nupic_output.NuPICFileOutput(["Machine_Temp_Sys_Failure_OUTPUT_CSV"])
        outputPlotFile = nupic_output.NuPICPlotOutput(["Machine_Temp_Sys_Failure_OUTPUT_PLOT"])
    elif(dataset == 1):
        model_par = twitter_model_params
        csv_path = "./data/Twitter_volume_GOOG.csv"
        outputCSVFile = nupic_output.NuPICFileOutput(["Twitter_Volume_Google_OUTPUT_CSV"])
        outputPlotFile = nupic_output.NuPICPlotOutput(["Twitter_Volume_Google_OUTPUT_PLOT"])
    else:
        print("No specified dataset, error will occur")
        model_params = None

    # create model
    model = createModel(model_par)

    #run model
    runModel(model, csv_path, outputCSVFile, outputPlotFile)



if __name__ == "__main__":
    # dataset = 0 --> machine
    # dataset = 1 --> twitter
    dataset = 1
    runDataset(dataset)
