#!/usr/bin/env python

"""
Importing Packages
"""
# general
import csv
import datetime
import nupic_anomaly_output as nupic_output
import argparse

# model parameters
from machine_model_params import MODEL_PARAMS as machine_model_params
from twitter_model_params import MODEL_PARAMS as twitter_model_params

# model
from nupic.frameworks.opf.model_factory import ModelFactory

# plotting
from nupic.data.inference_shifter import InferenceShifter


"""
Global variables
"""
DATE_FORMAT = "%m/%d/%Y %H:%M" # ex) 2/3/2001 21:45 


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
       
        # plotting prediction and anomaly detection
        plot_result = shifter.shift(result)
        plot_prediction = plot_result.inferences["multiStepBestPredictions"][1]
        plot_anomalyScore = plot_result.inferences["anomalyScore"]
        outputPlotFile.write(timestamp, value, plot_prediction, plot_anomalyScore)

        # output csv for anomaly detection
        prediction = result.inferences["multiStepBestPredictions"][1]
        anomalyScore = result.inferences["anomalyScore"]
        outputCSVFile.write(timestamp, value, prediction, anomalyScore)


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
        nupic_output.WINDOW = 22694
        nupic_output.ANOMALY_THRESHOLD = 0.97
        outputCSVFile = nupic_output.NuPICFileOutput("Machine_Temp_Sys_Failure_OUTPUT_ANOMALY_CSV")
        outputPlotFile = nupic_output.NuPICPlotOutput("Machine_Temp_Sys_Failure_OUTPUT_ANOMALY_PLOT")
    elif(dataset == 1):
        model_par = twitter_model_params
        csv_path = "./data/Twitter_volume_GOOG.csv"
        print(nupic_output.WINDOW)
        nupic_output.WINDOW = 15841
        print(nupic_output.WINDOW)
        outputCSVFile = nupic_output.NuPICFileOutput("Twitter_Volume_Google_OUTPUT_ANOMALY_CSV")
        outputPlotFile = nupic_output.NuPICPlotOutput("Twitter_Volume_Google_OUTPUT_ANOMALY_PLOT")
    else:
        print("No specified dataset, error will occur")
        model_params = None

    # create model
    model = createModel(model_par)

    #run model
    runModel(model, csv_path, outputCSVFile, outputPlotFile)

def create_parser():
    """
    Creates parser for command line inputs
    """
    parser = argparse.ArgumentParser(description='Parser for HTM streaming anomaly detection')

    # function to allow parsing of true/false boolean
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')       

    """
    Hierarchical Temporal Memory arguments
    """

    # arguments for both Spatial Pooler and Temporal Memory
    parser.add_argument('--dataset', type=int, default=0,
                        help='Determines dataset being used, where machine = 0 and twitter = 1; default=0')

    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    """
    Main program to run anomaly detection
    """

    # create parser
    args = create_parser()

    runDataset(args.dataset)
