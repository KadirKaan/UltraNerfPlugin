/* Copyright (c) 2012-2024 ImFusion GmbH, Munich, Germany. All rights reserved. */
#pragma once

#include <ImFusion/GUI/AlgorithmController.h>

#include <QtWidgets/QWidget>

class Ui_UltraNerfInferenceController;

namespace ImFusion
{
	class UltraNerfInferenceAlgorithm;

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

	protected:
		Ui_UltraNerfInferenceController *m_ui; ///< The actual GUI
		UltraNerfInferenceAlgorithm *m_alg;	   //< The algorithm instance
	};
}