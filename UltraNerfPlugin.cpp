#include "UltraNerfPlugin.h"

#include "UltraNerfFactory.h"

// Export free factory function to instantiate plugin
#ifdef WIN32
extern "C" __declspec(dllexport) ImFusion::ImFusionPlugin *createPlugin()
{
	return new ImFusion::UltraNerfPlugin;
}
#else
extern "C" ImFusion::ImFusionPlugin *createPlugin()
{
	return new ImFusion::UltraNerfPlugin;
}
#endif

namespace ImFusion
{
	UltraNerfPlugin::UltraNerfPlugin() {}

	UltraNerfPlugin::~UltraNerfPlugin() {}

	const ImFusion::AlgorithmFactory *UltraNerfPlugin::getAlgorithmFactory() { return new UltraNerfAlgorithmFactory; }

	const ImFusion::AlgorithmControllerFactory *UltraNerfPlugin::getAlgorithmControllerFactory()
	{
		return new UltraNerfControllerFactory;
	}
}