Final Year Project:

Project Proposal: Hybrid AI System for Gym Exercise Form Analysis Using Pose Estimation and Machine Learning

This project aims to develop an intelligent computer vision system that analyses human exercise form from video footage and provides real-time feedback on movement quality and potential injury risk. The system will use pose estimation (via MediaPipe) to extract 3D skeletal joint coordinates from exercise videos, then apply both rule-based biomechanics checks and machine learning classification to detect risky or incorrect movement patterns.
The project will initially focus on a single exercise type — the squat — due to its biomechanical complexity and availability of public data. The system will later be extended to other movements such as lunges or push-ups if feasible.

Academically, the work integrates core themes from:
    • Computer Vision & Imaging: pose extraction and motion analysis
    • Machine Learning & Neural Computation: risk classification and feature learning
    • Algorithms & Complexity: real-time computation and efficiency
    • Software Engineering: system design and integration of multiple components

    1. Implement pose estimation using MediaPipe to detect and track body joints in exercise videos.
    2. Compute joint angles (e.g., hip, knee, ankle, spine) and extract biomechanical features per frame or repetition.
    3. Design a rule-based analysis system to identify key form errors (e.g., “knee over toe”, “insufficient depth”, “spine curvature”).
    4. Build a machine learning classifier trained on these extracted features to automatically distinguish “safe” vs “risky” reps.
    5. Integrate both approaches into a hybrid feedback system that:
        ○ Uses rules for explainable feedback, and
        ○ Uses ML for adaptive risk scoring.
    6. Evaluate the system’s accuracy across multiple subjects and camera angles.
    7. Visualise results (e.g., highlighting incorrect joints, overlaying feedback text on video).

Is it worth exploring 3D pose estimation for camera-angle invariance, or should v1 assume a fixed side/45° view for simplicity?
Would you advise incorporating user-interface development (e.g., a demo app), or focusing purely on the analysis pipeline?
How many samples of one movement (e.g., squat) and from how many camera angles would you recommend collecting for training and validation, and can I get some examples from social media (e.g youtube)?



