/* Copyright (c) 2012-2024 ImFusion GmbH, Munich, Germany. All rights reserved. */
#pragma once

#include <ImFusion/Base/ImFusionPlugin.h>

namespace ImFusion
{
	class AlgorithmFactory;
	class AlgorithmControllerFactory;

	// See also the ExamplePlugin for further documentation
	class UltraNerfPlugin : public ImFusionPlugin
	{
	public:
		UltraNerfPlugin();
		virtual ~UltraNerfPlugin();
		virtual const AlgorithmFactory *getAlgorithmFactory();
		virtual const AlgorithmControllerFactory *getAlgorithmControllerFactory();
	};
}