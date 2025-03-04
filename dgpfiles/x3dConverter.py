import bpy

for obj in bpy.context.scene.objects:
    bpy.data.objects.remove(obj, do_unlink=True)

bpy.ops.wm.ply_import(filepath="/home/labuser/iron_tari/triangulated_cloud.ply")  # Update with path to ply

bpy.ops.object.editmode_toggle()
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.remove_doubles(threshold=0.002, use_unselected=True)
bpy.ops.mesh.delete_loose(use_faces=True)
bpy.ops.mesh.dissolve_degenerate(threshold=0.001)
bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='EDGE')
bpy.ops.mesh.select_all(action='DESELECT')
bpy.ops.mesh.edges_select_sharp()
bpy.ops.mesh.mark_seam(clear=False)
bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='FACE')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.001)
bpy.ops.export_scene.x3d(filepath='test.x3d', **{"use_selection": False, 'use_mesh_modifiers':True, 'use_normals': False, 'global_scale':0.001})
