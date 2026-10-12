# SPDX-FileCopyrightText: Copyright 2023-2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Abundance-GS: the extension is built / reused by abundance_gs/ext.py (was: prebuilt import or JIT via .build).
from abundance_gs import ext

_C = ext.load('gsplat_cuda')

__all__ = ["_C"]
