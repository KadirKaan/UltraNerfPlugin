#include <type_traits>
#include "UltraNerfFactory.h"

#include "UltraNerfAlgorithm.h"
#include "UltraNerfController.h"

namespace ImFusion
{
	UltraNerfAlgorithmFactory::UltraNerfAlgorithmFactory()
	{
		// register the UltraNerfAlgorithm
		registerAlgorithm<UltraNerfAlgorithm>("UltraNerf;UltraNerf algorithm");
	}

	AlgorithmController *UltraNerfControllerFactory::create(Algorithm *a) const
	{
		// register the UltraNerfController for the UltraNerfAlgorithm
		if (UltraNerfAlgorithm *alg = dynamic_cast<UltraNerfAlgorithm *>(a))
			return new UltraNerfController(alg);
		return 0;
	}
}