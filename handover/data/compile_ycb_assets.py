import os

ycb_classes = [
    '002_master_chef_can',
    '003_cracker_box',
    '004_sugar_box',
    '005_tomato_soup_can',
    '006_mustard_bottle',
    '007_tuna_fish_can',
    '008_pudding_box',
    '009_gelatin_box',
    '010_potted_meat_can',
    '011_banana',
    '019_pitcher_base',
    '021_bleach_cleanser',
    '024_bowl',
    '025_mug',
    '035_power_drill',
    '036_wood_block',
    '037_scissors',
    '040_large_marker',
    '051_large_clamp',
    '052_extra_large_clamp',
    '061_foam_brick',
]

urdf_str = \
"""<?xml version="1.0"?>
<robot name="model_normalized">
  <link name="world">
    <inertial>
      <mass value="0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
  <joint name="world_base1" type="prismatic">
    <parent link="world"/>
    <child link="base1"/>
    <origin rpy="0 0 0" xyz="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1000" upper="1000"/>
  </joint>
  <link name="base1">
    <inertial>
      <mass value="0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
  <joint name="base1_base2" type="prismatic">
    <parent link="base1"/>
    <child link="base2"/>
    <origin rpy="0 0 0" xyz="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1000" upper="1000"/>
  </joint>
  <link name="base2">
    <inertial>
      <mass value="0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
  <joint name="base2_base3" type="prismatic">
    <parent link="base2"/>
    <child link="base3"/>
    <origin rpy="0 0 0" xyz="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1000" upper="1000"/>
  </joint>
  <link name="base3">
    <inertial>
      <mass value="0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
  <joint name="base3_base_link" type="spherical">
    <parent link="base3"/>
    <child link="base_link"/>
    <origin rpy="0 0 0" xyz="0 0 0"/>
    <axis xyz="1 0 0"/>
  </joint>
  <link name="base_link">
    <contact>
      <lateral_friction value="0.9"/>
    </contact>
    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <mass value="0.5"/> 
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="model_normalized.obj" scale="1 1 1"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="model_normalized_convex.obj" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>
</robot>
"""

src_root = "../../../../OMG-Planner/data/objects"
trg_root = os.path.join(os.path.dirname(__file__), "assets")


def main():
  print('Compiling YCB assets')

  for x in ycb_classes:
    print('{}'.format(x))
    src_dir = os.path.join(src_root, x)
    trg_dir = os.path.join(trg_root, x)
    os.makedirs(trg_dir, exist_ok=True)

    obj_files = ("model_normalized.obj", "model_normalized_convex.obj",
                 "textured_simple.obj.mtl", "texture_map.png")
    for y in obj_files:
      src_obj = os.path.join(src_dir, y)
      trg_obj = os.path.join(trg_dir, y)
      if not os.path.isfile(trg_obj):
        os.symlink(src_obj, trg_obj)
      else:
        assert os.readlink(trg_obj) == src_obj
    
    trg_urdf = os.path.join(trg_dir, "model_normalized.urdf")
    if not os.path.isfile(trg_urdf):
      with open(trg_urdf, 'w') as f:
        f.write(urdf_str)
    else:
      with open(trg_urdf, 'r') as f:
        assert f.read() == urdf_str

  print('Done.')


if __name__ == '__main__':
  main()
