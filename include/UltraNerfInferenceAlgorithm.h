/* Copyright (c) 2012-2024 ImFusion GmbH, Munich, Germany. All rights reserved. */
#pragma once

#include <ImFusion/Base/Algorithm.h>
#include <memory>

namespace ImFusion
{
	class SharedImageSet;

	enum class BLINE_ORIGIN : int
	{
		TOP = 0,
		LEFT = 1,
		RIGHT = 2,
		BOTTOM = 3
	};
	struct Point
	{
		float x;
		float y;
		float z;
		Point(float x, float y, float z) : x(x), y(y), z(z) {};
	};
	class UltraNerfInferenceAlgorithm : public Algorithm
	{
	public:
		UltraNerfInferenceAlgorithm();
		~UltraNerfInferenceAlgorithm();

		void setModelPath(std::string modelPath) { this->model_path = model_path; }
		void setPoints(Point point_top, Point point_bottom)
		{
			this->point_pair.first = point_top;
			this->point_pair.second = point_bottom;
		};

		void setBlineOrigin(BLINE_ORIGIN origin) { this->bline_origin = origin; }

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
		std::string model_path = "";
		BLINE_ORIGIN bline_origin = BLINE_ORIGIN::TOP;
	};
};
