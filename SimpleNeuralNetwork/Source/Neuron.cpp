#include "stdafx.h"

#include "Neuron.h"

#include <cmath>


namespace NeuralNetworks
{
	double Neuron::m_eta = 0.15;
	double Neuron::m_alpha = 0.5;

	//---------------------------------------------------------------
	Neuron::Neuron(size_t numOutputs, size_t index) :
		m_outputWeights(numOutputs),
		m_index(index)
	{
		for (size_t i = 0; i < numOutputs; ++i)
		{
			m_outputWeights[i].m_weight = randomWeight();
		}
	}

	//---------------------------------------------------------------
	void Neuron::feedForward(const std::vector<Neuron>& prevLayer)
	{
		double sum = 0.0;

		// Sum the previous layer's outputs (which are our inputs)
		// Include the bias node from the previous layer.

		for (size_t n = 0; n < prevLayer.size(); ++n)
		{
			sum += prevLayer[n].getOutputValue() * prevLayer[n].m_outputWeights[m_index].m_weight;
		}

		m_outputValue = transferFunction(sum);
	}

	//---------------------------------------------------------------
	void Neuron::calculateOutputGradients(double targetValue)
	{
		double delta = targetValue - m_outputValue;
		m_gradient = delta * transferFunctionDerivative(m_outputValue);
	}

	//---------------------------------------------------------------
	void Neuron::calculateHiddenGradients(const std::vector<Neuron>& nextLayer)
	{
		double dow = sumDOW(nextLayer);
		m_gradient = dow * transferFunctionDerivative(m_outputValue);
	}

	//---------------------------------------------------------------
	void Neuron::updateInputWeights(std::vector<Neuron>& prevLayer)
	{
		// The weights to be updated are in the connection struct
		// in the neurons in the previous layer
		for (size_t n = 0; n < prevLayer.size() - 1; ++n)
		{
			Neuron& neuron = prevLayer[n];
			double oldDeltaWeight = neuron.m_outputWeights[m_index].m_deltaWeight;
			double newDeltaWeight =
				// Individual input, magnified by the gradient and train rate:
				m_eta * neuron.getOutputValue() * m_gradient
				// Add also momentum = fraction of the previous delta weight
				+ m_alpha * oldDeltaWeight;

			neuron.m_outputWeights[m_index].m_deltaWeight = newDeltaWeight;
			neuron.m_outputWeights[m_index].m_weight += newDeltaWeight;
		}
	}

	//---------------------------------------------------------------
	double Neuron::sumDOW(const std::vector<Neuron>& nextLayer) const
	{
		double sum = 0;

		// Sum our contributions of the errors at the nodes we feed
		for (size_t n = 0; n < nextLayer.size() - 1; ++n)
		{
			sum += m_outputWeights[n].m_weight * nextLayer[n].m_gradient;
		}

		return sum;
	}

	//---------------------------------------------------------------
	double Neuron::randomWeight()
	{
		static std::mt19937 rng;
		//rng.seed(std::random_device()());
		std::uniform_real_distribution<double> dist;

		return dist(rng);
	}

	//---------------------------------------------------------------
	double Neuron::transferFunction(double x)
	{
		// tanh - output range [-1.0 ... 1.0]
		return tanh(x);
	}

	//---------------------------------------------------------------
	double Neuron::transferFunctionDerivative(double x)
	{
		// tanh derivative approximate
		return 1 - x * x;
	}
}