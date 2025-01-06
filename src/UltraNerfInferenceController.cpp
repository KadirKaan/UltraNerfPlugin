#include "UltraNerfInferenceController.h"
#include "UltraNerfInferenceAlgorithm.h"

#include <ImFusion/Base/DataModel.h>
#include <ImFusion/Base/FactoryRegistry.h>
#include <ImFusion/Base/SharedImageSet.h>
#include <ImFusion/Core/Log.h>
#include <ImFusion/GUI/MainWindowBase.h>
#include <ImFusion/GUI/DisplayWidgetMulti.h>
#include <ImFusion/GUI/ImageView2D.h>
#include <MyCustomGlObject.h>
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
	}

	UltraNerfInferenceController::~UltraNerfInferenceController()
	{
		if (m_myInteractiveObject)
			m_disp->view2D()->removeObject(m_myInteractiveObject.get());
	}
	void UltraNerfInferenceController::onCompute()
	{
		m_alg->setPoints(Point(m_ui->xCoorTop->text().toFloat(), m_ui->yCoorTop->text().toFloat(), m_ui->zCoorTop->text().toFloat()),
						 Point(m_ui->xCoorBot->text().toFloat(), m_ui->yCoorBot->text().toFloat(), m_ui->zCoorBot->text().toFloat()));
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
	void UltraNerfInferenceController::onClickMouse()
	{
		// lifetime is managed by this controller
		m_myInteractiveObject = std::make_unique<InteractiveObject>(new MyCustomGlObject);
		m_disp->view2D()->addObject(m_myInteractiveObject.get());
		m_disp->requestUpdate();
	}

	void UltraNerfInferenceController::init()
	{
		m_ui = new Ui_UltraNerfInferenceController();
		m_ui->setupUi(this);
		// TODO
		// connect(m_ui->pushButtonRunModel, SIGNAL(clicked()), this, SLOT(onCompute()));
		connect(m_ui->pushButtonLoadModel, SIGNAL(clicked()), this, SLOT(onLoadModel()));
		connect(m_ui->pushButtonRunModel, SIGNAL(clicked()), this, SLOT(onClickMouse()));
	}
}