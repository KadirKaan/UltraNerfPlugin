#include "UltraNerfInferenceController.h"

#include "UltraNerfInferenceAlgorithm.h"

#include <ImFusion/Base/DataModel.h>
#include <ImFusion/Base/FactoryRegistry.h>
#include <ImFusion/Base/SharedImageSet.h>
#include <ImFusion/Core/Log.h>
#include <ImFusion/GUI/MainWindowBase.h>

#include "ui_UltraNerfInferenceController.h"

// The following sets the log category for this file to "UltraNerfInferenceController"
#undef IMFUSION_LOG_DEFAULT_CATEGORY
#define IMFUSION_LOG_DEFAULT_CATEGORY "UltraNerfInferenceController"

// This class implements the GUI controller for the UltraNerfInferenceAlgorithm.
namespace ImFusion
{

	UltraNerfInferenceController::UltraNerfInferenceController(UltraNerfInferenceAlgorithm *algorithm)
		: AlgorithmController(algorithm), m_alg(algorithm)
	{
		m_ui = new Ui_UltraNerfInferenceController();
		m_ui->setupUi(this);
		// TODO
		connect(m_ui->pushButtonRunModel, SIGNAL(clicked()), this, SLOT(onCompute()));
		connect(m_ui->pushButtonLoadModel, SIGNAL(clicked()), this, SLOT(onLoadModel()));
	}

	void UltraNerfInferenceController::onCompute()
	{
		m_alg->setCoordinates(m_ui->xCoordinate->value(), m_ui->yCoordinate->value(), m_ui->zCoordinate->value());
		// Call compute on the algorithm
		m_alg->compute();
		m_main->dataModel()->add(m_alg->takeOutput());
	}
	void UltraNerfInferenceController::onLoadModel()
	{
		m_alg->setModelPath(m_ui->modelPath->text().toStdString());
		// Call compute on the algorithm
		m_alg->loadModel();
		m_main->dataModel()->add(m_alg->takeOutput());
	}

	UltraNerfInferenceController::~UltraNerfInferenceController()
	{
		delete m_ui;
	}

	void UltraNerfInferenceController::init() {}
}