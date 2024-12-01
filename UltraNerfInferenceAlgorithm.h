/* Copyright (c) 2012-2024 ImFusion GmbH, Munich, Germany. All rights reserved. */
#pragma once

#include <ImFusion/Base/Algorithm.h>
#include <UltraNerfTorch/include/NeRFModel.h>
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
		void setCoordinates(float x, float y, float z)
		{
			this->xCoordinate = x;
			this->yCoordinate = y;
			this->zCoordinate = z;
		}

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
		float xCoordinate = 0;
		float yCoordinate = 0;
		float zCoordinate = 0;
		NeRFModel model = NeRFModel();
		std::string modelPath = "";
	};
}