#pragma once

#include "Connection.h"

#include <vector>
#include <random>


namespace Neuron
{

class Neuron
{
	public:
		Neuron(size_t numOutputs, size_t index);

		double getOutputValue() const { return m_outputValue; }
		void setOutputValue(double outputValue) { m_outputValue = outputValue; }
	
		void feedForward(const std::vector<Neuron>& prevLayer);
		void calculateOutputGradients(double targetValue);
		void calculateHiddenGradients(const std::vector<Neuron>& nextLayer);
		void updateInputWeights(std::vector<Neuron>& prevLayer);

	private:
		static double randomWeight();
		static double transferFunction(double x);
		static double transferFunctionDerivative(double x);

		double sumDOW(const std::vector<Neuron>& nextLayer) const;

		static double m_eta;
		static double m_alpha;

		size_t m_index;
		double m_outputValue;
		double m_gradient;
		std::vector<Connection> m_outputWeights;
};

}