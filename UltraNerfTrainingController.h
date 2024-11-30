/* Copyright (c) 2012-2024 ImFusion GmbH, Munich, Germany. All rights reserved. */
#pragma once

#include <ImFusion/GUI/AlgorithmController.h>

#include <QtWidgets/QWidget>

class Ui_UltraNerfTrainingController;

namespace ImFusion
{
	class UltraNerfTrainingAlgorithm;

	// This class implements the GUI controller for the UltraNerfTrainingAlgorithm.
	class UltraNerfTrainingController : public QWidget, public AlgorithmController
	{
		Q_OBJECT

	public:
		// Constructor with the algorithm instance
		UltraNerfTrainingController(UltraNerfTrainingAlgorithm *algorithm);

		// Destructor
		virtual ~UltraNerfTrainingController();

		// Initializes the widget
		void init();

	public slots:
		void onCompute();

	protected:
		Ui_UltraNerfTrainingController *m_ui; ///< The actual GUI
		UltraNerfTrainingAlgorithm *m_alg;	  //< The algorithm instance
	};
}