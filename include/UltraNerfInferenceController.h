/* Copyright (c) 2012-2024 ImFusion GmbH, Munich, Germany. All rights reserved. */
#pragma once

#include <ImFusion/GUI/AlgorithmController.h>
#include <ImFusion/GL/GlRectangleBillboard.h>
#include <QtWidgets/QWidget>
#include <ImFusion/GUI/DisplayWidgetMulti.h>
#include <ImFusion/GUI/DefaultAlgorithmController.h>
class Ui_UltraNerfInferenceController;

namespace ImFusion
{
	class UltraNerfInferenceAlgorithm;
	class InteractiveObject;

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
		void onClickMouse();

	protected:
		Ui_UltraNerfInferenceController *m_ui; ///< The actual GUI
		UltraNerfInferenceAlgorithm *m_alg;	   //< The algorithm instance
		std::unique_ptr<InteractiveObject> m_myInteractiveObject;
	};
}