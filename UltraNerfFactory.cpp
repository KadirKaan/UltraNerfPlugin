#include <type_traits>
#include "UltraNerfFactory.h"

#include "UltraNerfTrainingAlgorithm.h"
#include "UltraNerfTrainingController.h"

#include "UltraNerfInferenceAlgorithm.h"
#include "UltraNerfInferenceController.h"
namespace ImFusion
{
	UltraNerfAlgorithmFactory::UltraNerfAlgorithmFactory()
	{
		// register the UltraNerfTrainAlgorithm
		registerAlgorithm<UltraNerfTrainingAlgorithm>("UltraNerf;UltraNerf Training Algorithm");
		registerAlgorithm<UltraNerfInferenceAlgorithm>("UltraNerf;UltraNerf Inference Algorithm");
	}

	AlgorithmController *UltraNerfControllerFactory::create(Algorithm *a) const
	{
		// register the UltraNerfController for the UltraNerfTrainAlgorithm
		if (UltraNerfTrainingAlgorithm *alg = dynamic_cast<UltraNerfTrainingAlgorithm *>(a))
			return new UltraNerfTrainingController(alg);
		// TODO: register inference as well
		if (UltraNerfInferenceAlgorithm *alg = dynamic_cast<UltraNerfInferenceAlgorithm *>(a))
			return new UltraNerfInferenceController(alg);
		return 0;
	}
}