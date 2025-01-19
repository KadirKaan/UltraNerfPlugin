/* Copyright (c) 2012-2024 ImFusion GmbH, Munich, Germany. All rights reserved. */
#pragma once

#include <ImFusion/GUI/AlgorithmController.h>
#include <QtWidgets/QWidget>
#include <ImFusion/GUI/DisplayWidgetMulti.h>
#include <ImFusion/GUI/AnnotationModel.h>
#include <ImFusion/GL/GlRectangle.h>

class Ui_UltraNerfInferenceController;

namespace ImFusion
{
	class UltraNerfInferenceAlgorithm;
	class InteractiveObject;
	class AnnotationModel;
	class GlRectangle;
	// This class implements the GUI controller for the UltraNerfInferenceAlgorithm.
	class UltraNerfInferenceController : public QWidget, public AlgorithmController
	{
		Q_OBJECT

	public:
		// Constructor with the algorithm instance
		UltraNerfInferenceController(UltraNerfInferenceAlgorithm *algorithm);

		// Destructor
		virtual ~UltraNerfInferenceController();

		// Initializes the widget
		void init();

	public slots:
		void onCompute();
		void onLoadModel();
		void on_get_points();

	protected:
		Ui_UltraNerfInferenceController *m_ui; ///< The actual GUI
		UltraNerfInferenceAlgorithm *m_alg;	   //< The algorithm instance
		bool drawing = false;

	private:
		void copy_points(const GlRectangle *rectangle);
	};
}