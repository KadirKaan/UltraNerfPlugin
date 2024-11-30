#include "UltraNerfInferenceAlgorithm.h"

#include <ImFusion/Base/DataList.h>
#include <ImFusion/Base/ImageProcessing.h>
#include <ImFusion/Base/MemImage.h>
#include <ImFusion/Base/Pose.h>
#include <ImFusion/Base/SharedImage.h>
#include <ImFusion/Base/SharedImageSet.h>
#include <ImFusion/Base/Utils/Images.h>
#include <ImFusion/Core/Log.h>

// The following sets the log category for this file to "UltraNerf"
#undef IMFUSION_LOG_DEFAULT_CATEGORY
#define IMFUSION_LOG_DEFAULT_CATEGORY "UltraNerf"

namespace ImFusion
{
	UltraNerfInferenceAlgorithm::UltraNerfInferenceAlgorithm()
	{
	}

	UltraNerfInferenceAlgorithm::~UltraNerfInferenceAlgorithm()
	{
	}

	bool UltraNerfInferenceAlgorithm::createCompatible(const DataList &data, Algorithm **a)
	{
		// check requirements to create the algorithm. In this case, we can generate whenever we want to (maybe input model file)
		// requirements are met, create the algorithm if asked
		if (a)
		{
			*a = new UltraNerfInferenceAlgorithm();
		}
		return true;
	}

	// This function does all of the work of this class
	void UltraNerfInferenceAlgorithm::compute()
	{
		// set generic error status until we have finished
		m_status = static_cast<int>(Status::Error);
		m_imgOut = std::make_unique<SharedImageSet>();
		// set algorithm status to success
		m_status = static_cast<int>(Status::Success);
	}

	void UltraNerfInferenceAlgorithm::loadModel()
	{
	}
	OwningDataList UltraNerfInferenceAlgorithm::takeOutput()
	{
		// if we have produced some output, add it to the list
		return OwningDataList(std::move(m_imgOut));
	}

	void UltraNerfInferenceAlgorithm::configure(const Properties *p)
	{
		// this method restores our members when a workspace file is loaded
		if (p == nullptr)
			return;

		p->param("xCoordinate", xCoordinate);
		p->param("yCoordinate", yCoordinate);
		p->param("zCoordinate", zCoordinate);
		p->param("alphaAngle", alphaAngle);
		p->param("omegaAngle", omegaAngle);
		p->param("modelPath", modelPath);

		signalParametersChanged.emitSignal();
	}

	void UltraNerfInferenceAlgorithm::configuration(Properties *p) const
	{
		// this method is necessary to store our settings in a workspace file
		if (p == nullptr)
			return;

		p->setParam("xCoordinate", xCoordinate, 0.f);
		p->setParam("yCoordinate", yCoordinate, 0.f);
		p->setParam("zCoordinate", zCoordinate, 0.f);
		p->setParam("alphaAngle", alphaAngle, 0.f);
		p->setParam("omegaAngle", omegaAngle, 0.f);
	}
}