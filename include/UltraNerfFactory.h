/* Copyright (c) 2012-2024 ImFusion GmbH, Munich, Germany. All rights reserved. */
#pragma once

#include <ImFusion/Base/AlgorithmControllerFactory.h>
#include <ImFusion/Base/AlgorithmFactory.h>

namespace ImFusion
{
	class Algorithm;

	// See also the ExamplePlugin for further documentation

	class UltraNerfAlgorithmFactory : public AlgorithmFactory
	{
	public:
		UltraNerfAlgorithmFactory();
	};

	class UltraNerfControllerFactory : public AlgorithmControllerFactory
	{
	public:
		virtual AlgorithmController *create(Algorithm *a) const;
	};
}