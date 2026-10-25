# P2+CBAM Windows entry points

Reuse the exact Python/CUDA environment of the previous P2 run. Do not reinstall/upgrade dependencies.

- `00_check.bat`: data, environment, initialization, forward preflight; never trains.
- `01_train.bat`: new P2+CBAM run, 50 epochs, original frozen P2 hyperparameters.
- `03_check_resume.bat`: checks resumability without training.
- `02_resume.bat`: continue the default run's full last.pt, never restart silently.
- `04_control_p2_optional.bat`: OPTIONAL new P2 control; never launched automatically.

Detailed Chinese instructions: `docs/P2_CBAM_TRAINING.md`.

Results: `runs/helmet_p2_cbam_isolated_640_windows`. Existing datasets and weights stay in place.
If the wrong interpreter is detected, use the original training terminal or set YOLO_PYTHON to the original python.exe path.
