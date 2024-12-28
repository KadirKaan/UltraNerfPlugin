/* Copyright (c) 2012-2024 ImFusion GmbH, Munich, Germany. All rights reserved. */
#pragma once

#include <ImFusion/Base/Algorithm.h>
#include <UltraNerfTorch/include/UltraNeRFRenderer.h>
#include <UltraNerfTorch/include/NeRFUtils.h>
#include <memory>

namespace ImFusion
{
	class SharedImageSet;
	class UltraNerfInferenceAlgorithm : public Algorithm
	{
	public:
		UltraNerfInferenceAlgorithm();
		~UltraNerfInferenceAlgorithm();

		void setModelPath(std::string modelPath) { this->modelPath = modelPath; }
		void setPoints(Point point_top, Point point_bottom)
		{
			this->point_pair.first = point_top;
			this->point_pair.second = point_bottom;
		};

		// \name	Methods implementing the algorithm interface
		//\{
		// Factory method to check for applicability or to create the algorithm
		static bool createCompatible(const DataList &data, Algorithm **a = 0);

		void compute() override;
		void loadModel();

		// If new data was created, make it available here
		OwningDataList takeOutput() override;

		/// \name	Methods implementing the Configurable interface
		//\{
		void configure(const Properties *p) override;
		void configuration(Properties *p) const override;
		//\}

	private:
		std::unique_ptr<SharedImageSet> m_imgOut; ///< Output image after processing
		// TODO: derive these
		std::pair<Point, Point> point_pair = std::make_pair(Point(0, 0, 0), Point(0, 0, 0));
		UltraNeRFRenderer renderer;
		std::string modelPath = "";
		torch::Tensor generate_rays();
	};
};
