#include "UltraNerfInferenceAlgorithm.h"

#include <ImFusion/Base/DataList.h>
#include <ImFusion/Base/ImageProcessing.h>
#include <ImFusion/Base/MemImage.h>
#include <ImFusion/Base/Pose.h>
#include <ImFusion/Base/SharedImage.h>
#include <ImFusion/Base/SharedImageSet.h>
#include <ImFusion/Base/Utils/Images.h>
#include <ImFusion/Core/Log.h>
#include <UltraNerfTorch/include/NeRFUtils.h>

// The following sets the log category for this file to "UltraNerf"
#undef IMFUSION_LOG_DEFAULT_CATEGORY
#define IMFUSION_LOG_DEFAULT_CATEGORY "UltraNerf"

namespace ImFusion
{
	UltraNerfInferenceAlgorithm::UltraNerfInferenceAlgorithm()
	{
		// TODO: remove hard coded values
		float H = 512.0;
		float W = 256.0;
		float scale = 0.001;
		float probe_depth = 140.0 * scale;
		float probe_width = 80.0 * scale;
		float sh = probe_depth / H;
		float sw = probe_width / W;
		torch::manual_seed(0);
		NeRFModel model = NeRFModel(NeRFUtils::get_device());
		this->renderer_ptr = std::make_unique<UltraNeRFRenderer>(model, int(H), int(W), sw, sh);
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
		// TODO: enums in QT
		torch::Tensor rays = NeRFUtils::generate_rays(this->point_pair.first, this->point_pair.second, 512, BLINE_ORIGIN::TOP);
		torch::Tensor rays_o = rays.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)});
		torch::Tensor rays_d = rays.index({torch::indexing::Slice(), torch::indexing::Slice(3, 6)});
		torch::Dict<std::string, torch::Tensor> render_results = renderer_ptr.get()->render_nerf(std::make_pair(rays_o, rays_d), std::nullopt);
		torch::Tensor output = renderer_ptr.get()->get_output_data(render_results);
		// TODO: torch tensor to imfusion image set conversion, hard to implement with no option to check
		m_imgOut = std::make_unique<SharedImageSet>();
		// set algorithm status to success
		m_status = static_cast<int>(Status::Success);
	}

	void UltraNerfInferenceAlgorithm::loadModel()
	{
		std::string modelPath = this->model_path;
		renderer_ptr.get()->get_model_ptr()->load_weights(modelPath);
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

		p->param("xCoorTop", this->point_pair.first.x);
		p->param("yCoorTop", this->point_pair.first.y);
		p->param("zCoorTop", this->point_pair.first.z);
		p->param("xCoorBot", this->point_pair.second.x);
		p->param("yCoorBot", this->point_pair.second.y);
		p->param("zCoorBot", this->point_pair.second.z);
		p->param("modelPath", model_path);

		signalParametersChanged.emitSignal();
	}

	void UltraNerfInferenceAlgorithm::configuration(Properties *p) const
	{
		// this method is necessary to store our settings in a workspace file
		if (p == nullptr)
			return;
		p->setParam("modelPath", model_path, "");
	}
};