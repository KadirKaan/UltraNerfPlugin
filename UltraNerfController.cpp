#include "UltraNerfController.h"

#include "UltraNerfAlgorithm.h"

#include <ImFusion/Base/DataModel.h>
#include <ImFusion/Base/FactoryRegistry.h>
#include <ImFusion/Base/SharedImageSet.h>
#include <ImFusion/Core/Log.h>
#include <ImFusion/GUI/MainWindowBase.h>

#include "ui_UltraNerfController.h"

// The following sets the log category for this file to "UltraNerfController"
#undef IMFUSION_LOG_DEFAULT_CATEGORY
#define IMFUSION_LOG_DEFAULT_CATEGORY "UltraNerfController"

// This class implements the GUI controller for the UltraNerfAlgorithm.
namespace ImFusion
{

	UltraNerfController::UltraNerfController(UltraNerfAlgorithm *algorithm)
		: AlgorithmController(algorithm), m_alg(algorithm)
	{
		m_ui = new Ui_UltraNerfController();
		m_ui->setupUi(this);
		connect(m_ui->pushButtonApply, SIGNAL(clicked()), this, SLOT(onCompute()));
	}

	void UltraNerfController::onCompute()
	{
		m_alg->setFactor(m_ui->spinBoxFactor->value());
		// Call compute on the algorithm
		m_alg->compute();
		m_main->dataModel()->add(m_alg->takeOutput());
	}

	UltraNerfController::~UltraNerfController()
	{
		delete m_ui;
	}

	void UltraNerfController::init() {}
}