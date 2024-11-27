#include "UltraNerfController.h"

#include "UltraNerfAlgorithm.h"

#include <ImFusion/Base/DataModel.h>
#include <ImFusion/Base/FactoryRegistry.h>
#include <ImFusion/Base/SharedImageSet.h>
#include <ImFusion/Core/Log.h>
#include <ImFusion/CT/GUI/XRay2D3DRegistrationAlgorithmController.h>
#include <ImFusion/CT/GUI/XRay2D3DRegistrationInitializationController.h>
#include <ImFusion/CT/GUI/XRay2D3DRegistrationInitializationKeyPointsController.h>
#include <ImFusion/CT/XRay2D3DRegistrationAlgorithm.h>
#include <ImFusion/GUI/MainWindowBase.h>

#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QPushButton>

// The following sets the log category for this file to "UltraNerfController"
#undef IMFUSION_LOG_DEFAULT_CATEGORY
#define IMFUSION_LOG_DEFAULT_CATEGORY "UltraNerfController"

// This class implements the GUI controller for the UltraNerfAlgorithm.
namespace ImFusion
{

	UltraNerfController::UltraNerfController(UltraNerfAlgorithm *algorithm)
		: AlgorithmController(algorithm), m_alg(algorithm)
	{
		// Adds a button with which to launch the algorithm
		m_computeButton = new QPushButton("Compute projections, launch ultra-nerf");
		this->setLayout(new QHBoxLayout);
		this->layout()->addWidget(m_computeButton);
		connect(m_computeButton, SIGNAL(clicked()), this, SLOT(onCompute()));
	}

	void UltraNerfController::onCompute()
	{
		// Disable the button created in constructor.
		if (m_computeButton)
			m_computeButton->setEnabled(false);

		// Call compute on the algorithm
		m_alg->compute();
		m_main->dataModel()->add(m_alg->takeOutput());
		m_computeButton->setEnabled(true);
	}

	UltraNerfController::~UltraNerfController() {}

	void UltraNerfController::init() {}
}