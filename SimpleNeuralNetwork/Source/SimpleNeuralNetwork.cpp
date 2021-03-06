// SimpleNeuralNetwork.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include "Network.h"
#include "TrainingData.h"

#include <iostream>
#include <fstream>

using namespace NeuralNetworks;

void showVectorValues(const std::string& message, const std::vector<double>& values)
{
	std::cout << message << " ";
	for (size_t i = 0; i < values.size(); ++i)
	{
		std::cout << values[i] << " ";
	}
}

int main()
{
	/*std::ofstream myfile;
	myfile.open("data.txt");

	myfile << "topology: 2 4 1" << std::endl;
	for (size_t i = 0; i <= 2000; ++i)
	{
		size_t n1 = static_cast<size_t>(2.0 * rand() / double(RAND_MAX));
		size_t n2 = static_cast<size_t>(2.0 * rand() / double(RAND_MAX));
		size_t t = n1 ^ n2;

		myfile << "in: " << n1 << ".0 " << n2 << ".0 " << std::endl;
		myfile << "out: " << t << ".0" << std::endl;
	}

	myfile.close();
	return 0;*/

	TrainingData trainingData("data.txt");

	std::vector<unsigned int> topology;
	trainingData.getTopology(topology);

	Network network(topology);

	std::vector<double> inputValues, targetValues, resultValues;
	size_t trainingPass = 0;

	while (!trainingData.isEof())
	{
		++trainingPass;
		std::cout << std::endl << "Pass " << trainingPass;

		// Get new input data and feed it forward
		if (trainingData.getNextInputs(inputValues) != topology[0])
		{
			break;
		}

		showVectorValues(": Inputs:", inputValues);
		network.feedForward(inputValues);

		// Collect the net's results
		network.getResults(resultValues);
		showVectorValues("Outputs: ", resultValues);

		// Train the net with what the outputs should have been
		trainingData.getTargetOutputs(targetValues);
		showVectorValues("Targets: ", targetValues);

		network.backPropogate(targetValues);

		// Report how well the training is working, averaged over recent iterations
		std::cout << "Net recent average error: " << network.getRecentAverageError() << std::endl;
	}

	std::cout << std::endl << "Done" << std::endl;

    return 0;
}