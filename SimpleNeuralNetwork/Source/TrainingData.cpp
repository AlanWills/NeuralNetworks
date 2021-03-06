#include "stdafx.h"

#include "TrainingData.h"

#include <sstream>


namespace NeuralNetworks
{
	//---------------------------------------------------------------
	TrainingData::TrainingData(const std::string& filePath)
	{
		m_trainingDataFile.open(filePath);
	}

	//---------------------------------------------------------------
	void TrainingData::getTopology(std::vector<unsigned int>& topology)
	{
		std::string line;
		std::string label;

		std::getline(m_trainingDataFile, line);
		std::stringstream ss(line);
		ss >> label;

		if (isEof() || label.compare("topology:") != 0)
		{
			abort();
		}
		while (!ss.eof())
		{
			unsigned int n;
			ss >> n;

			topology.push_back(n);
		}
	}

	//---------------------------------------------------------------
	size_t TrainingData::getNextInputs(std::vector<double>& inputValues)
	{
		inputValues.clear();

		std::string line;
		getline(m_trainingDataFile, line);
		std::stringstream ss(line);

		std::string label;
		ss >> label;
		if (label.compare("in:") == 0)
		{
			double oneValue;
			while (ss >> oneValue)
			{
				inputValues.push_back(oneValue);
			}
		}

		return inputValues.size();
	}

	//---------------------------------------------------------------
	size_t TrainingData::getTargetOutputs(std::vector<double>& targetOutputValues)
	{
		targetOutputValues.clear();

		std::string line;
		getline(m_trainingDataFile, line);
		std::stringstream ss(line);

		std::string label;
		ss >> label;
		if (label.compare("out:") == 0)
		{
			double oneValue;
			while (ss >> oneValue)
			{
				targetOutputValues.push_back(oneValue);
			}
		}

		return targetOutputValues.size();
	}
}