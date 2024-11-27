/* Copyright (c) 2012-2024 ImFusion GmbH, Munich, Germany. All rights reserved. */
#pragma once

#include <ImFusion/Base/Algorithm.h>

#include <memory>

namespace ImFusion
{
	class SharedImageSet;

	class UltraNerfAlgorithm : public Algorithm
	{
	public:
		// Creates the algorithm instance with an image
		UltraNerfAlgorithm(SharedImageSet *img);
		~UltraNerfAlgorithm();

		/// Set downsampling factor
		void setFactor(int factor) { m_factor = factor; }

		// \name	Methods implementing the algorithm interface
		//\{
		// Factory method to check for applicability or to create the algorithm
		static bool createCompatible(const DataList &data, Algorithm **a = 0);

		void compute() override;

		// If new data was created, make it available here
		OwningDataList takeOutput() override;

		/// \name	Methods implementing the Configurable interface
		//\{
		void configure(const Properties *p) override;
		void configuration(Properties *p) const override;
		//\}

	private:
	private:
		SharedImageSet *m_imgIn = nullptr;		  ///< Input image to process
		std::unique_ptr<SharedImageSet> m_imgOut; ///< Output image after processing
		int m_factor = 2;						  ///< Downsampling factor
	};
}