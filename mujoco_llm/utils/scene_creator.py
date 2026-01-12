import os
import re
import xml.etree.ElementTree as ET
import xml.dom.minidom
from typing import List, Dict

class SceneCreator:
    @staticmethod
    def absolutize_asset_paths(tree: ET.ElementTree, base_path: str) -> ET.ElementTree:
        """
        XML 내부 file attribute를 base_path 기준의 절대경로로 변환.
        """
        base_dir = os.path.dirname(os.path.abspath(base_path))
        for element in tree.findall(".//*[@file]"):
            filepath = element.get("file")
            if not os.path.isabs(filepath):
                abs_filepath = os.path.join(base_dir, filepath)
                element.set("file", os.path.normpath(abs_filepath))
        return tree
    
    @staticmethod
    def load_xml_tree_with_absolute_paths(xml_path: str) -> ET.ElementTree:
        """
        xml_path 파일을 파싱하고, file attribute를 모두 절대경로로 변환하여 반환
        """
        tree = ET.parse(xml_path)
        tree = SceneCreator.absolutize_asset_paths(tree, xml_path)
        return tree

    @staticmethod
    def add_robot_to_scene(scene_tree: ET.ElementTree, robot_tree: ET.ElementTree):
        """
        로봇 xml의 <worldbody>, <asset>, <actuator>, <default>를 scene에 추가
        """
        root = scene_tree.getroot()
        worldbody = scene_tree.find("worldbody")
        asset = scene_tree.find("asset")
        if asset is None:
            asset = ET.Element("asset")
            root.insert(0, asset)
        actuator = scene_tree.find("actuator")
        if actuator is None:
            actuator = ET.Element("actuator")
            root.append(actuator)

        # default section
        robot_default = robot_tree.find("default")
        if robot_default is not None:
            exist_default = root.find("default")
            if exist_default is not None:
                root.remove(exist_default)
            option_tag = root.find("option")
            if option_tag is not None:
                children = list(root)
                option_idx = children.index(option_tag)
                root.insert(option_idx + 1, robot_default)
            else:
                root.insert(0, robot_default)

        # visual 병합 추가!
        robot_visual = robot_tree.find("visual")
        if robot_visual is not None:
            exist_visual = root.find("visual")
            if exist_visual is not None:
                # 기존 visual의 rgba를 찾는다
                exist_rgba = exist_visual.find("rgba")
                robot_rgba = robot_visual.find("rgba")
                if robot_rgba is not None:
                    # robot rgba의 속성을 기존 rgba에 merge
                    if exist_rgba is not None:
                        for k, v in robot_rgba.attrib.items():
                            exist_rgba.set(k, v)
                    else:
                        # 기존 visual에 rgba가 없으면 통째로 추가
                        exist_visual.append(robot_rgba)
                # robot_visual의 나머지 child는 append
                for child in robot_visual:
                    if child.tag != "rgba":
                        exist_visual.append(child)
            else:
                # visual이 없으면 통째로 추가
                root.insert(1, robot_visual)

        # worldbody
        robot_worldbody = robot_tree.find("worldbody")
        for body in robot_worldbody.findall("body"):
            worldbody.append(body)

        # asset
        robot_asset = robot_tree.find("asset")
        if robot_asset is not None:
            for item in robot_asset:
                asset.append(item)
                
        # sensor 병합
        robot_sensor = robot_tree.find("sensor")
        if robot_sensor is not None:
            exist_sensor = root.find("sensor")
            if exist_sensor is not None:
                # 기존 sensor에 없는 센서만 추가 (중복 체크)
                exist_sensor_names = {s.get('name') for s in exist_sensor}
                for child in robot_sensor:
                    if child.get('name') not in exist_sensor_names:
                        exist_sensor.append(child)
            else:
                # sensor가 아예 없으면 통째로 추가
                root.append(robot_sensor)

        # actuator
        robot_actuator = robot_tree.find("actuator")
        if robot_actuator is not None:
            for actuator_elem in robot_actuator:
                actuator.append(actuator_elem)

    @staticmethod
    def add_object_to_scene(scene_tree: ET.ElementTree, obj_path: str, obj_name: str, obj_pos: str):
        """
        단일 객체 XML을 scene에 추가 (body, asset, actuator)
        """
        obj_tree = SceneCreator.load_xml_tree_with_absolute_paths(obj_path)
        worldbody = scene_tree.find("worldbody")
        asset = scene_tree.find("asset")
        actuator = scene_tree.find("actuator")

        # 새로운 body
        new_body = ET.Element("body", attrib={"name": obj_name, "pos": obj_pos})

        # free joint
        unique_joint_name = obj_name + "_joint"
        free_joint = ET.Element("joint", attrib={"type": "free", "name": unique_joint_name})
        new_body.append(free_joint)

        # geom, site
        for geom in obj_tree.findall(".//geom"):
            new_body.append(geom)
        for site in obj_tree.findall(".//site"):
            site_name = site.get("name")
            if site_name:
                site.set("name", f"{obj_name}_{site_name}")
            new_body.append(site)
        worldbody.append(new_body)

        # asset
        obj_asset = obj_tree.find("asset")
        if obj_asset is not None:
            for item in obj_asset:
                asset.append(item)
        # actuator
        obj_actuator = obj_tree.find("actuator")
        if obj_actuator is not None:
            for actuator_elem in obj_actuator:
                actuator.append(actuator_elem)

    @staticmethod
    def pretty_print_xml(xml_string: str) -> str:
        dom = xml.dom.minidom.parseString(xml_string)
        pretty_xml = dom.toprettyxml(indent="  ")
        pretty_xml = re.sub(r'\n\s*\n+', '\n', pretty_xml)
        return pretty_xml

    @staticmethod
    def build_mjcf_scene(
        base_env_path: str,
        robot_path: str,
        objects_to_spawn: List[Dict[str, str]],
        save_xml: bool = False,
    ) -> str:
        """
        base_env_path, robot_path, objects_to_spawn 정보를 받아
        최종적으로 조합된 MJCF XML string 반환
        """
        # base 환경
        scene_tree = SceneCreator.load_xml_tree_with_absolute_paths(base_env_path)
        # 로봇 추가
        robot_tree = SceneCreator.load_xml_tree_with_absolute_paths(robot_path)
        SceneCreator.add_robot_to_scene(scene_tree, robot_tree)
        # 오브젝트 추가
        for obj in objects_to_spawn:
            SceneCreator.add_object_to_scene(scene_tree, obj["path"], obj["name"], obj["pos"])
        for geom in scene_tree.findall(".//geom"):
            if geom.get("name") == "floor":
                geom.set("friction", "2.5 0.01 0.01") # slide, roll, spin
        # XML string 반환
        root = scene_tree.getroot()
        xml_str = ET.tostring(root, encoding='unicode')
        pretty_xml = SceneCreator.pretty_print_xml(xml_str)

        if save_xml:
            with open("scene.xml", "w", encoding="utf-8") as f:
                f.write(pretty_xml)
            print("saved scene.xml")

        return pretty_xml