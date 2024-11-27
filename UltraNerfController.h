/* Copyright (c) 2012-2024 ImFusion GmbH, Munich, Germany. All rights reserved. */
#pragma once

#include <ImFusion/GUI/AlgorithmController.h>

#include <QtWidgets/QWidget>

class Ui_UltraNerfController;

namespace ImFusion
{
	class UltraNerfAlgorithm;

	// This class implements the GUI controller for the UltraNerfAlgorithm.
	class UltraNerfController : public QWidget, public AlgorithmController
	{
		Q_OBJECT

	public:
		// Constructor with the algorithm instance
		UltraNerfController(UltraNerfAlgorithm *algorithm);

		// Destructor
		virtual ~UltraNerfController();

		// Initializes the widget
		void init();

	public slots:
		void onCompute();

	protected:
		Ui_UltraNerfController *m_ui; ///< The actual GUI
		UltraNerfAlgorithm *m_alg;	  //< The algorithm instance
	};
}