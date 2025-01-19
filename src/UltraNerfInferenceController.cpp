#include "UltraNerfInferenceController.h"
#include "UltraNerfInferenceAlgorithm.h"

#include <ImFusion/Base/DataModel.h>
#include <ImFusion/Base/FactoryRegistry.h>
#include <ImFusion/Base/SharedImageSet.h>
#include <ImFusion/Core/Log.h>
#include <ImFusion/GUI/MainWindowBase.h>
#include <ImFusion/GUI/DisplayWidgetMulti.h>
#include <ImFusion/GUI/ImageView2D.h>
#include <iostream>
#include <ImFusion/GL/GlSliceView.h>
#include <ImFusion/Core/Mat.h>
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
	}
	void UltraNerfInferenceController::onCompute()
	{
		m_alg->setPoints(Point(m_ui->xCoorTop->text().toFloat(), m_ui->yCoorTop->text().toFloat(), m_ui->zCoorTop->text().toFloat()),
						 Point(m_ui->xCoorBot->text().toFloat(), m_ui->yCoorBot->text().toFloat(), m_ui->zCoorBot->text().toFloat()));
		m_alg->setBlineOrigin(BLINE_ORIGIN(m_ui->originEnum));
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
	void UltraNerfInferenceController::init()
	{
		m_ui = new Ui_UltraNerfInferenceController();
		m_ui->setupUi(this);
		connect(m_ui->pushButtonRunModel, SIGNAL(clicked()), this, SLOT(onCompute()));
		connect(m_ui->pushButtonLoadModel, SIGNAL(clicked()), this, SLOT(onLoadModel()));
		connect(m_ui->pushButtonGetPoints, SIGNAL(clicked()), this, SLOT(on_get_points()));
	}

	void UltraNerfInferenceController::on_get_points()
	{
		// get the first image
		// todo: make it smarter
		auto annotations = m_main->annotationModel()->getAnnotations(m_main->dataModel()->getAll().getFirst());

		// get the rectangle annotation, maybe smarter later on?

		auto it = std::find_if(annotations.begin(), annotations.end(), [](const auto &annotation)
							   { return annotation->gl()->typeName() == "GlRectangle"; });

		if (it == annotations.end())
		{
			return;
		}
		auto rectangle_annotation = *it;
		// get the rectangle
		auto rectangle = dynamic_cast<GlRectangle *>(rectangle_annotation->gl());
		copy_points(rectangle);
	}

	void UltraNerfInferenceController::copy_points(const GlRectangle *rectangle)
	{
		auto corners = rectangle->rectangleCorners();
		m_ui->xCoorTop->setText(QString::number(corners[0][0]));
		m_ui->yCoorTop->setText(QString::number(corners[0][1]));
		m_ui->zCoorTop->setText(QString::number(corners[0][2]));
		m_ui->xCoorBot->setText(QString::number(corners[3][0]));
		m_ui->yCoorBot->setText(QString::number(corners[3][1]));
		m_ui->zCoorBot->setText(QString::number(corners[3][2]));
	}
}