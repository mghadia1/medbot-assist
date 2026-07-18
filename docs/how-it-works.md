# How MedBot Assist works

## Current milestone boundary

This milestone connects the existing MedBot Vision classical detector to the existing SurgiArm planar planner through defined data messages and coordinate transforms. It is a cross-platform integration test, not ROS 2, Gazebo, a physical robot, or clinical software.

MedBot Vision itself detects connected components but does not classify instrument names. MedBot Assist now supports two clearly separated adapters. The oracle adapter matches detections to simulator ground-truth centroids and records `label_source=simulator_ground_truth_oracle`. The learned adapter crops each detected component, runs a four-class PyTorch CNN, and records `label_source=synthetic_cnn_prediction`.

The CNN is trained only on generated scalpel, forceps, clamp, and retractor silhouettes with randomized scale, angle, position, brightness, and noise. It is a real learned prediction within that synthetic domain, but it is not evidence of classification on photographs or clinical tools.

## Pipeline

1. MedBot Vision generates a labeled synthetic tray image.
2. Its threshold/component/PCA detector estimates each component's centroid and angle.
3. The selected adapter either uses oracle matching for pipeline debugging or predicts a class and softmax confidence with the synthetic CNN.
4. The safety gate finds exactly one requested name, checks detection age, confidence, and image bounds, and rejects unsafe input.
5. A fitted homography maps the pixel centroid to the flat tray plane.
6. A rigid 2D transform maps the tray point into the robot-base frame.
7. SurgiArm checks inverse kinematics and collision-aware paths, then executes its planar pick-and-place state machine.
8. The integration records success or a safe rejection/planning failure.

## Homography

A homography is a 3-by-3 projective matrix for points on a plane. Four corner correspondences provide eight independent equations when the last matrix scale is fixed. After multiplication, the result is divided by its homogeneous third coordinate.

This is appropriate for a flat tray under a fixed camera model. It does not recover arbitrary 3D depth, handle tools lifted above the plane, or replace camera intrinsic calibration and distortion correction.

## Safety gates

The coordinator refuses to plan when:

- the requested name is missing or duplicated;
- the component count does not match the controlled synthetic scene;
- the detection is stale or timestamped in the future;
- confidence is below the configured threshold;
- the centroid lies outside the calibrated image;
- inverse kinematics or collision-aware planning fails.

Rejecting a request is a valid result. A safe integration system should not force an answer when its assumptions fail.

## Later milestones

The flagship still needs a learned multi-class tool model, measured calibration with a camera model, overlapping-tool handling, ROS 2 messages/actions, TF2, MoveIt, Gazebo physics, ros2_control, task verification, and supported-Linux end-to-end evaluation.
