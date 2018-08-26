#pragma once

#include "NeuronDllExport.h"
#include "Neuron.h"


namespace Neuron
{

class Neuron;

class Network
{
	public:
		typedef std::vector<Neuron> Layer;

    NeuronDllExport Network();

		double getRecentAverageError() const { return m_error; }

    void initializeTopology(const std::vector<size_t>& topology);
		void feedForward(const std::vector<double>& inputValues);
		void backPropogate(const std::vector<double>& targetValues);
		void getResults(std::vector<double>& resultValues) const;

	private:
		std::vector<Layer> m_layers;	// m_layers[layer_num][neuron_num];
		double m_error;
};

}