import numpy as np
import easysim
from easysim.constants import GeometryType

class PrimitiveObject:
    def __init__(self, cfg, scene):
        self._cfg = cfg
        self._scene = scene
        
        self._bodies = {}
        for i in range(self._cfg.ENV.NUM_PRIMITIVE_OBJECTS):
            body = self._create_body(
                'primitive_object_{:02d}'.format(i),
                self._cfg.ENV.PRIMITIVE_OBJECT_MASS,
                self._cfg.ENV.PRIMITIVE_OBJECT_BASE_POSITION[i],
                [0.1, 0.9, 0.1, 1.0],
                self._cfg.ENV.COLLISION_FILTER_PRIMITIVE_OBJECT
            )
            self._scene.add_body(body)
            self._bodies[i] = body
        
        if self._cfg.ENV.DISPLAY_TARGET:
            body = self._create_body(
                "target",
                0,
                [
                    self._cfg.ENV.TARGET_POSITION_X,
                    self._cfg.ENV.TARGET_POSITION_Y,
                    self._cfg.ENV.PRIMITIVE_OBJECT_SIZE / 2
                ],
                [1.0, 0.0, 0.0, 0.7],
                0
            )
            self._scene.add_body(body)
    
    @property
    def bodies(self):
        return self._bodies
    
    def _create_body(self, name, mass, position, color, collision_filter):
        body = easysim.Body()
        body.name = name
        if self._cfg.ENV.PRIMITIVE_OBJECT_TYPE == 'Box':
            body.geometry_type = GeometryType.BOX
        elif self._cfg.ENV.PRIMITIVE_OBJECT_TYPE == 'Sphere':
            body.geometry_type = GeometryType.SPHERE
        else:
            raise ValueError(f'{self._cfg.ENV.PRIMITIVE_OBJECT_TYPE} is not supported by PrimitiveObject class.')
        half_size = self._cfg.ENV.PRIMITIVE_OBJECT_SIZE / 2
        if body.geometry_type == GeometryType.BOX:
            body.box_half_extent = [half_size, half_size, half_size]
        else:
            body.sphere_radius = half_size
        body.base_mass = mass
        body.initial_base_position = [position + [0, 0, 0, 1]]
        body.link_color = [color]
        body.link_collision_filter = [collision_filter]
        return body