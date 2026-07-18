# Integration milestone results

Run date: July 16, 2026

## Demonstration

The seed-7 demonstration requested the scalpel. The classical detector found four components. The adapter matched the scalpel component to simulator truth and recorded the class source as an oracle rather than a prediction.

- detected scalpel centroid: `(147.369, 124.239)` pixels;
- geometry-association confidence: 0.9315;
- transformed planar robot target: `(1.3318, 0.4428)`;
- manipulation result: complete;
- planning iterations: 596;
- RRT-required segments: 2;
- final object-to-bin error: 0 in the ideal planar attachment model.

The zero placement error is an exact-simulation artifact. The class label is simulator truth. Neither is a real-robot or learned-recognition result.

## Twenty randomized integration trials

The seed-101 evaluation varied the positions and angles of four generated tools and cycled through requested tool names.

- successful end-to-end trials: 15;
- safely rejected trials: 5;
- planning failures after acceptance: 0;
- success rate: 75%;
- mean requested-tool centroid error: 5.8302 pixels;
- mean robot-target error for accepted trials: 0.0070 planar units.

All five non-successes were confidence-gate rejections. Their centroid errors were approximately 8.00–8.44 pixels, producing geometry-association confidence slightly below the 0.80 policy threshold. Four involved forceps and one involved the clamp. The system did not send these uncertain targets to the arm.

This is desirable safety behavior, but it also exposes a weakness: the hand-designed association confidence is not statistically calibrated. A later model must learn tool classes and calibrate confidence on held-out data instead of treating centroid distance as recognition certainty.

## Explicitly tested rejections

Automated tests verify rejection for:

- missing requested name;
- stale detection;
- confidence below policy threshold;
- centroid outside the calibrated image;
- component-count mismatch before message creation.

## Limitations

- simulator ground truth supplies the tool names;
- the detector assumes separated bright silhouettes;
- the homography uses ideal corner correspondences with no lens distortion;
- the tray is planar and fixed;
- the arm is a 2D exact-kinematics model with perfect grasp attachment;
- no learned multi-class model, camera stream, ROS 2, TF2, MoveIt, Gazebo, ros2_control, physics, or hardware.

The full flagship requires new evidence after each of those boundaries is replaced.

## Learned synthetic classifier milestone

The classifier dataset contains 192 generated training-pool crops and 48 separately generated held-out test crops across four balanced classes. After a deterministic 80/20 split of the training pool, 154 samples fit the model and 38 selected the checkpoint.

- selected epoch: 23 of 24;
- validation accuracy: 100%;
- held-out crop accuracy: 100% (48/48);
- mean held-out softmax confidence: 0.9549;
- held-out confusion matrix: 12 correct samples on each class diagonal and zero off-diagonal errors.

This is a small synthetic test from the same generator family, not real-world generalization evidence.

When the learned checkpoint replaced oracle names in the twenty randomized full-scene integration trials:

- component class predictions: 80/80 correct;
- requested tasks accepted and completed: 11/20;
- safely rejected for confidence below 0.80: 9/20;
- accepted-task planning failures: 0;
- mean robot-target error for accepted tasks: 0.0083 planar units.

The classifier was correct on all components but less confident on full-scene detector crops than on the isolated held-out crop set. All five clamp requests were rejected; four additional scalpel/retractor requests were rejected. This is evidence of domain shift and confidence-calibration weakness, so the safety threshold was not lowered merely to increase the success rate.
