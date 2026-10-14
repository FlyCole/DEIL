import os

ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')

# Calibration files are machine-specific inputs and are excluded from Git.
# T_WC maps head-camera coordinates to world; T_EC maps wrist camera to EEF.
T_WCdir = os.path.join(ASSETS_DIR, 'front_calibration/T_WC_head.npy')
T_ECdir = os.path.join(ASSETS_DIR, 'front_calibration/T_EC_wrist.npy')
T_WC_side_dir = os.path.join(ASSETS_DIR, 'side_calibration/T_WC_side.npy')

HOME_j = [0.049340534216351255, -0.48337516757645044, -0.0011477664886754466, 
          -1.8819137264050732, -0.0058514479984469545, 1.4036752802779704, 0.8671925621611901]
