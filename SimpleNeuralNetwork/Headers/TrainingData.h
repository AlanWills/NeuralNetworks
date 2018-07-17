#pragma once

#include <filesystem>
#include <fstream>


namespace NeuralNetworks
{

class TrainingData
{
	public:
		TrainingData(const std::string& filePath);

		bool isEof() const { return m_trainingDataFile.eof(); }
		
		void getTopology(std::vector<unsigned int>& topology);
		size_t getNextInputs(std::vector<double>& nextInputs);
		size_t getTargetOutputs(std::vector<double>& targetOutputs);

	private:
		std::ifstream m_trainingDataFile;
};

}