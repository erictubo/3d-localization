import subprocess
import sys

blender_path = "/Applications/Blender.app/Contents/MacOS/Blender"


# def render_database():

#     script_path = sys.path[0] + "/blender/render_database.py"

#     command = [
#         blender_path,
#         "--background",
#         "--python", script_path,
#         "--",
#     ]

#     subprocess.run(command)


def render_query(
        blend_file: str,
        target_name: str,
        render_dir: str,
        intrinsics_file: str,
        poses_file: str,
        quaternion_first: bool,
        limit: int,
    ) -> None:

    script_path = sys.path[0] + "/blender/render_query.py"

    if render_dir[-1] != '/':
        render_dir += '/'

    command = [
        blender_path,
        "--background",
        "--python", script_path,
        "--",
        "--blend_file", blend_file,
        "--target_name", target_name,
        "--render_dir", render_dir,
        "--intrinsics_file", intrinsics_file,
        "--poses_file", poses_file,
        "--limit", str(limit),
    ]

    if quaternion_first:
        command.append("--quaternion_first")

    subprocess.run(command)