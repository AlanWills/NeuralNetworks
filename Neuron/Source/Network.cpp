#include "stdafx.h"

#include "Network.h"


namespace Neuron
{
	//---------------------------------------------------------------
	Network::Network() :
    m_layers(),
    m_error(0)
	{
	}

  //---------------------------------------------------------------
  void Network::initializeTopology(const std::vector<size_t>& topology)
  {
    size_t numLayers = topology.size();
    m_layers.clear();
    m_layers.reserve(numLayers);

    for (size_t layerNum = 0; layerNum < numLayers; ++layerNum)
    {
      m_layers.push_back(Layer());
      size_t numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

      // We have made a new layer, let's now fill it with neurons
      // + one bias neuron
      for (size_t neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)
      {
        m_layers.back().push_back(Neuron(numOutputs, neuronNum));
      }

      // Force the bias node's output value to 1.0.   It's the last neuron created above.
      m_layers.back().back().setOutputValue(1.0);
    }
  } 

	//---------------------------------------------------------------
	void Network::feedForward(const std::vector<double>& inputValues)
	{
		for (size_t i = 0; i < inputValues.size(); ++i)
		{
			m_layers[0][i].setOutputValue(inputValues[i]);
		}

		// Forward propogate
		for (size_t layerNum = 1; layerNum < m_layers.size(); ++layerNum)
		{
			Layer& prevLayer = m_layers[layerNum - 1];
			for (size_t i = 0; i < m_layers[layerNum].size() - 1; ++i)
			{
				m_layers[layerNum][i].feedForward(prevLayer);
			}
		}
	}

	//---------------------------------------------------------------
	void Network::backPropogate(const std::vector<double>& targetValues)
	{
		// Calculate overall net error (RMS of output neuron errors)
		std::vector<Neuron>& outputLayer = m_layers.back();
		m_error = 0;

		for (size_t n = 0; n < targetValues.size(); ++n)
		{
			double delta = targetValues[n] - outputLayer[n].getOutputValue();
			m_error += delta * delta;
		}

		m_error /= (outputLayer.size() - 1);
		m_error = sqrt(m_error);

		// Calculate output layer gradients
		for (size_t n = 0; n < outputLayer.size() - 1; ++n)
		{
			outputLayer[n].calculateOutputGradients(targetValues[n]);
		}
		
		// Calculate gradients on hidden layers
		for (size_t layerNum = m_layers.size() - 2; layerNum > 0; --layerNum)
		{
			Layer& hiddenLayer = m_layers[layerNum];
			Layer& nextLayer = m_layers[layerNum + 1];

			for (size_t n = 0; n < hiddenLayer.size(); ++n)
			{
				hiddenLayer[n].calculateHiddenGradients(nextLayer);
			}
		}

		// For all layers from outputs to first hidden layer,
		// update connection weights
		for (size_t layerNum = m_layers.size() - 1; layerNum > 0; --layerNum)
		{
			Layer& layer = m_layers[layerNum];
			Layer& prevLayer = m_layers[layerNum - 1];

			for (size_t n = 0; n < layer.size() - 1; ++n)
			{
				layer[n].updateInputWeights(prevLayer);
			}
		}
	}

	//---------------------------------------------------------------
	void Network::getResults(std::vector<double>& resultValues) const
	{
		resultValues.clear();

		for (size_t n = 0; n < m_layers.back().size() - 1; ++n)
		{
			resultValues.push_back(m_layers.back()[n].getOutputValue());
		}
	}
}