import trimesh
import numpy as np

def generate_flat_plane_x3d(output_file):
    # A simple plane (2.5D surface) like what a laser scanner would produce
    # 4 vertices, 2 triangles
    verts = [
        [-0.25, -0.25, 0.0],
        [ 0.25, -0.25, 0.0],
        [ 0.25,  0.25, 0.0],
        [-0.25,  0.25, 0.0]
    ]
    faces = [
        [0, 1, 2],
        [0, 2, 3]
    ]
    
    verts_str = " ".join([f"{v[0]} {v[1]} {v[2]}" for v in verts])
    
    # UV mapping (0 to 1)
    uvs = [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ]
    uvs_str = " ".join([f"{u[0]} {u[1]}" for u in uvs])
    
    faces_str = " ".join([f"{f[0]} {f[1]} {f[2]} -1" for f in faces])
    
    x3d_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE X3D PUBLIC "ISO//Web3D//DTD X3D 3.3//EN" "http://www.web3d.org/specifications/x3d-3.3.dtd">
<X3D profile="Interchange" version="3.3">
  <Scene>
    <Transform>
      <Shape>
        <IndexedFaceSet coordIndex="{faces_str}" texCoordIndex="{faces_str}">
          <Coordinate point="{verts_str}"/>
          <TextureCoordinate point="{uvs_str}"/>
        </IndexedFaceSet>
        <Appearance>
          <Material diffuseColor="0.8 0.8 0.8"/>
        </Appearance>
      </Shape>
    </Transform>
  </Scene>
</X3D>
"""
    with open(output_file, 'w') as f:
        f.write(x3d_content)

def generate_flat_plane_dae(output_file):
    # Export the same plane to DAE for Noether
    verts = [
        [-0.25, -0.25, 0.0],
        [ 0.25, -0.25, 0.0],
        [ 0.25,  0.25, 0.0],
        [-0.25,  0.25, 0.0]
    ]
    faces = [
        [0, 1, 2],
        [0, 2, 3]
    ]
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(output_file)

if __name__ == "__main__":
    generate_flat_plane_x3d("/home/user/software_ws/ROS-Scanning/dgpfiles/flat_plate.x3d")
    generate_flat_plane_dae("/home/user/software_ws/ROS-Scanning/dgpfiles/flat_plate.dae")
