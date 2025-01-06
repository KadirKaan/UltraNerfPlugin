#include "UltraNerfTrainingController.h"

#include "UltraNerfTrainingAlgorithm.h"

#include <ImFusion/Base/DataModel.h>
#include <ImFusion/Base/FactoryRegistry.h>
#include <ImFusion/Base/SharedImageSet.h>
#include <ImFusion/Core/Log.h>
#include <ImFusion/GUI/MainWindowBase.h>

#include "ui_UltraNerfTrainingController.h"

// The following sets the log category for this file to "UltraNerfTrainingController"
#undef IMFUSION_LOG_DEFAULT_CATEGORY
#define IMFUSION_LOG_DEFAULT_CATEGORY "UltraNerfTrainingController"

// This class implements the GUI controller for the UltraNerfTrainingAlgorithm.
namespace ImFusion
{

	UltraNerfTrainingController::UltraNerfTrainingController(UltraNerfTrainingAlgorithm *algorithm)
		: AlgorithmController(algorithm), m_alg(algorithm)
	{
		m_ui = new Ui_UltraNerfTrainingController();
		m_ui->setupUi(this);
		// TODO
		connect(m_ui->pushButtonApply, SIGNAL(clicked()), this, SLOT(onCompute()));
	}

	void UltraNerfTrainingController::onCompute()
	{
		m_alg->setFactor(m_ui->spinBoxFactor->value());
		// Call compute on the algorithm
		m_alg->compute();
		m_main->dataModel()->add(m_alg->takeOutput());
	}

	UltraNerfTrainingController::~UltraNerfTrainingController()
	{
		delete m_ui;
	}

	void UltraNerfTrainingController::init() {}
}