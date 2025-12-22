#!/usr/bin/env python3
"""
Script to process Orca hand URDFs:
1. Convert package:// mesh paths to relative paths
2. Add 6-DoF wrist joints using the add_wrist_dof.py script
"""
import os
import sys
import re

# Add the parent directory to path for importing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from lxml import etree


def convert_mesh_paths(input_urdf: str, output_urdf: str):
    """Convert package:// paths to relative paths."""
    with open(input_urdf, 'r') as f:
        content = f.read()
    
    # The final URDF files will be in orca_hand/ directory
    # The meshes are in orca_hand/meshes/urdf/{left,right}/{visual,collision}/
    # So relative path is meshes/urdf/
    
    # Replace package://orcahand_description/assets/urdf/left/
    # with meshes/urdf/left/
    content = re.sub(
        r'package://orcahand_description/assets/urdf/(left|right)/(visual|collision)/',
        r'meshes/urdf/\1/\2/',
        content
    )
    
    with open(output_urdf, 'w') as f:
        f.write(content)
    
    print(f"Converted mesh paths: {input_urdf} -> {output_urdf}")


def remove_world_link_and_tower(input_urdf: str, output_urdf: str, side: str = 'left'):
    """
    Remove world link, tower link, and fixed joints that connect them.
    Make the palm link the root of the kinematic chain.
    This is necessary because add_wrist_dof.py expects the first joint to be fixed
    and connect to the base link.
    """
    lxml_parser = etree.XMLParser(remove_comments=True, remove_blank_text=True)
    tree = etree.parse(input_urdf, parser=lxml_parser)
    robot_elem = tree.getroot()
    
    palm_link = f"{side}_palm"
    tower_link = f"{side}_tower"
    wrist_joint = f"{side}_wrist"
    
    # Find elements to remove
    elements_to_remove = []
    
    for link_elem in robot_elem.findall("link"):
        link_name = link_elem.attrib.get("name", "")
        if link_name in ["world", tower_link, f"world2{side}_tower_fixed_jointbody", 
                         f"{side}_wrist_jointbody"]:
            elements_to_remove.append(link_elem)
    
    for joint_elem in robot_elem.findall("joint"):
        joint_name = joint_elem.attrib.get("name", "")
        if joint_name in [f"world2{side}_tower_fixed", f"world2{side}_tower_fixed_offset",
                          wrist_joint, f"{side}_wrist_offset"]:
            elements_to_remove.append(joint_elem)
    
    for elem in elements_to_remove:
        robot_elem.remove(elem)
    
    # Add a new world link and fixed joint to palm
    world_link = etree.Element("link", name="world")
    inertial = etree.SubElement(world_link, "inertial")
    etree.SubElement(inertial, "origin", xyz="0.0 0.0 0.0", rpy="0.0 0.0 0.0")
    etree.SubElement(inertial, "mass", value="0.0")
    etree.SubElement(inertial, "inertia", ixx="0.0", iyy="0.0", izz="0.0", ixy="0", ixz="0", iyz="0")
    
    robot_elem.insert(0, world_link)
    
    # Add fixed joint from world to palm
    fixed_joint = etree.Element("joint", name="world_to_palm", type="fixed")
    etree.SubElement(fixed_joint, "parent", link="world")
    etree.SubElement(fixed_joint, "child", link=palm_link)
    etree.SubElement(fixed_joint, "origin", xyz="0.0 0.0 0.0", rpy="0.0 0.0 0.0")
    
    robot_elem.insert(1, fixed_joint)
    
    tree.write(output_urdf, pretty_print=True, xml_declaration=True, encoding="utf-8")
    print(f"Simplified URDF structure: {input_urdf} -> {output_urdf}")


def add_6dof_wrist(input_urdf: str, output_urdf: str, base_link: str, is_left: bool = True):
    """Add 6-DoF wrist joints using the logic from add_wrist_dof.py"""
    from dataclasses import dataclass
    from typing import Tuple, Dict
    
    @dataclass
    class Dof:
        joint_type: str
        axis: Tuple[int, int, int]
        joint_range: Tuple[float, float] = (-2.0, 2.0)
        stiffness: int = 1
        reflect: bool = False
        effort: int = 100
        velocity: int = 5

    _FOREARM_DOFS: Dict[str, Dof] = {
        "forearm_tx": Dof(joint_type="prismatic", axis=(1, 0, 0)),
        "forearm_ty": Dof(joint_type="prismatic", axis=(0, 1, 0)),
        "forearm_tz": Dof(joint_type="prismatic", axis=(0, 0, 1)),
        "forearm_roll": Dof(joint_type="revolute", axis=(0, 0, 1), joint_range=(-6.2, 6.2)),
        "forearm_pitch": Dof(joint_type="revolute", axis=(1, 0, 0), joint_range=(-6.2, 6.2)),
        "forearm_yaw": Dof(joint_type="revolute", axis=(0, -1, 0), joint_range=(-6.2, 6.2)),
    }

    side_prefix = "L_" if is_left else "R_"
    dof_choices = ["forearm_yaw", "forearm_pitch", "forearm_roll", 
                   "forearm_tz", "forearm_ty", "forearm_tx"]
    
    lxml_parser = etree.XMLParser(remove_comments=True, remove_blank_text=True)
    tree = etree.parse(input_urdf, parser=lxml_parser)
    robot_elem = tree.getroot()
    
    # Find the first fixed joint (world_to_palm)
    joint_elem = None
    for j in robot_elem.findall("joint"):
        if j.attrib.get("type") == "fixed":
            joint_elem = j
            break
    
    assert joint_elem is not None, "No fixed joint found"
    
    curr_base_name = base_link
    new_joint_names = []
    
    for dof_choice in dof_choices:
        dof = _FOREARM_DOFS[dof_choice]
        new_link_name = f"{side_prefix}{dof_choice}_link"
        
        # Create new dummy link
        new_link = etree.Element("link", name=new_link_name)
        mass_elem = etree.SubElement(new_link, "mass", value="0.01")
        robot_elem.insert(1, new_link)
        
        # Modify the child of prev_joint_elem
        child_elem = joint_elem.find("child")
        child_elem.attrib["link"] = new_link_name
        
        # Create new joint
        new_joint_name = f"{new_link_name}_joint"
        new_joint = etree.Element("joint", name=new_joint_name, type=dof.joint_type)
        etree.SubElement(new_joint, "parent", link=new_link_name)
        etree.SubElement(new_joint, "child", link=curr_base_name)
        
        axis = dof.axis
        axis_str = " ".join(str(x) for x in axis)
        etree.SubElement(new_joint, "axis", xyz=axis_str)
        
        lower, upper = dof.joint_range
        etree.SubElement(new_joint, "limit", lower=str(lower), upper=str(upper),
                        effort=str(dof.effort), velocity=str(dof.velocity))
        
        robot_elem.insert(1, new_joint)
        new_joint_names.append(new_joint_name)
        curr_base_name = new_link_name
    
    tree.write(output_urdf, pretty_print=True, xml_declaration=True, encoding="utf-8")
    print(f"Added 6-DoF wrist joints: {output_urdf}")
    print(f"New joint names: {new_joint_names}")
    return new_joint_names


def get_joint_names(urdf_path: str):
    """Get all non-fixed joint names from URDF"""
    lxml_parser = etree.XMLParser(remove_comments=True, remove_blank_text=True)
    tree = etree.parse(urdf_path, parser=lxml_parser)
    robot_elem = tree.getroot()
    
    joint_names = []
    for joint_elem in robot_elem.findall("joint"):
        if joint_elem.attrib.get("type", "fixed") != "fixed":
            joint_names.append(joint_elem.attrib["name"])
    
    return joint_names


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models", "urdf")
    
    for side in ["left", "right"]:
        is_left = (side == "left")
        palm_link = f"{side}_palm"
        
        # Input and output paths
        raw_urdf = os.path.join(models_dir, f"orcahand_{side}.urdf")
        fixed_paths_urdf = os.path.join(script_dir, f"orcahand_{side}_fixed.urdf")
        simplified_urdf = os.path.join(script_dir, f"orcahand_{side}_simple.urdf")
        final_urdf = os.path.join(script_dir, f"orcahand_{side}_6dof.urdf")
        
        # Step 1: Convert mesh paths
        convert_mesh_paths(raw_urdf, fixed_paths_urdf)
        
        # Step 2: Remove world/tower links
        remove_world_link_and_tower(fixed_paths_urdf, simplified_urdf, side)
        
        # Step 3: Add 6-DoF wrist
        add_6dof_wrist(simplified_urdf, final_urdf, palm_link, is_left)
        
        # Print joint names for reference
        joint_names = get_joint_names(final_urdf)
        print(f"\n{side.upper()} hand joint names:")
        for jn in joint_names:
            print(f"  '{jn}',")
        
        # Clean up intermediate files
        os.remove(fixed_paths_urdf)
        os.remove(simplified_urdf)
        
    print("\n✅ Successfully created orcahand_left_6dof.urdf and orcahand_right_6dof.urdf")


if __name__ == "__main__":
    main()

