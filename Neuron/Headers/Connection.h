#pragma once


namespace Neuron
{

class Connection
{
  public:
    Connection(double weight = 0) :
      m_weight(weight)
    {
    }

    double getWeight() const { return m_weight; }
    void setWeight(double weight) { m_weight = weight; }
    void incrementWeight(double modifier) { m_weight += modifier; }

    double getDeltaWeight() const { return m_deltaWeight; }
    void setDeltaWeight(double weight) { m_deltaWeight = weight; }

  private:
    double m_weight;
	  double m_deltaWeight;
};

}