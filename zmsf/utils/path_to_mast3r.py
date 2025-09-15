# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/


import sys
import os.path as path
HERE_PATH = path.normpath(path.dirname(__file__))
MAST3R_REPO_PATH = path.normpath(path.join(HERE_PATH, '../mast3r'))
MAST3R_LIB_PATH = path.join(MAST3R_REPO_PATH, 'mast3r')
# check the presence of mast3r directory in repo to be sure its cloned
if path.isdir(MAST3R_LIB_PATH):
    # workaround for sibling import
    sys.path.insert(0, MAST3R_REPO_PATH)
else:
    raise ImportError(f"mast3r is not initialized, could not find: {MAST3R_LIB_PATH}.\n "
                      "Did you forget to run 'git submodule update --init --recursive' ?")
