# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the BSD 3-Clause License [see LICENSE for details].

import easysim
import os


class Table:
    def __init__(self, cfg, scene):
        self._cfg = cfg
        self._scene = scene

        body = easysim.Body()
        body.name = "table"
        body.geometry_type = easysim.GeometryType.BOX
        body.box_half_extent = (
            self._cfg.ENV.TABLE_LENGTH / 2,
            self._cfg.ENV.TABLE_WIDTH / 2,
            self._cfg.ENV.TABLE_HEIGHT / 2
        )
        body.base_mass = 0.0
        body.initial_base_position = (
            (self._cfg.ENV.TABLE_X_OFFSET, 0.0, -self._cfg.ENV.TABLE_HEIGHT / 2) + (0, 0, 0, 1)
        )
        if self._cfg.SIM.SIMULATOR == "bullet":
            body.link_color = [(0.95, 0.95, 0.95, 1.0)]
        body.link_collision_filter = [self._cfg.ENV.COLLISION_FILTER_TABLE]
        self._scene.add_body(body)
        self._body = body

    @property
    def body(self):
        return self._body
