#pragma once

#if BUILDING_NEURON_DLL
#define NeuronDllExport __declspec(dllexport)
#else
#define NeuronDllExport __declspec(dllimport)
#endif