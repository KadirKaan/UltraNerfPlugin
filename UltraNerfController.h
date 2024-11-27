/* Copyright (c) 2012-2024 ImFusion GmbH, Munich, Germany. All rights reserved. */
#pragma once

#include <ImFusion/GUI/AlgorithmController.h>

#include <QtWidgets/QWidget>

class QPushButton;

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
		UltraNerfAlgorithm *m_alg;				//< The algorithm instance
		QPushButton *m_computeButton = nullptr; //< Button that is clicked to launch onCompute
	};
}