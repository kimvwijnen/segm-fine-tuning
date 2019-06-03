import pathlib
import requests

import numpy as np
import scipy.io as scio
import typing

import PIL.Image

from collections import namedtuple

import xml.etree.ElementTree as ElementTree
from concurrent.futures.process import ProcessPoolExecutor
import hashlib
import portalocker
import io
from concurrent.futures.thread import ThreadPoolExecutor


this_file_path = pathlib.Path(__file__).absolute().parent

def resolve_path(path: pathlib.Path):
    if (path / "redirect").exists():
        with open(path / "redirect", "r") as f:
            path = pathlib.Path(f.read().strip())
        return resolve_path(path)
    return path


cache_path = resolve_path(this_file_path / "cache")
annotation_path = resolve_path(this_file_path / "Annotation")


def load_image(url):
    hasher = hashlib.md5()
    hasher.update(url.encode())
    hex_dig = hasher.hexdigest()
    
    path = cache_path / hex_dig
    try:
        if not path.exists():
            with portalocker.Lock(path, "wb") as f:
                response = requests.get(url)
                response.raise_for_status()
                f.write(response.content)
    except requests.HTTPError:
        return None
    
    with open(path, "rb") as f:
        try:
            pil_image = PIL.Image.open(io.BytesIO(f.read()))
            return np.array(pil_image.getdata()).reshape(pil_image.size[1], pil_image.size[0], 3)
        except OSError:
            return None


Image = namedtuple("Image", ["name", "class_name", "url", "annotation", "load"])
Rect = namedtuple("Rect", ["minx", "maxx", "miny", "maxy"])


def load_image_list():
    __filehandle: typing.TextIO = None # :type self.__filehandle: typing.TextIO 
    
    def load_annotation_for_image(image: Image):
        filepath = annotation_path / image.class_name / f"{image.name}.xml"
        if not filepath.exists():
            return []
        elements = ElementTree.parse(filepath)
        rects = []
        for obj in elements.findall("./object"):
            if image.class_name == obj.find("./name").text.strip():
                rects.append(Rect(
                    int(obj.find("./bndbox/xmin").text),
                    int(obj.find("./bndbox/xmax").text),
                    int(obj.find("./bndbox/ymin").text),
                    int(obj.find("./bndbox/ymax").text),
                ))
        return rects
    
    class DelayedFileList:
        def __init__(self, filename):
            self.__loader = ThreadPoolExecutor(4)
    
            self.__filehandle = open(filename, "r", encoding="iso-8859-1") # :type self.__filehandle: typing.TextIO
            self.__lines = 0
            self.__thousand_offsets = [0]
            line = self.__filehandle.readline()
            while line:
                if line.strip():
                    self.__lines += 1
                    if self.__lines % 1000 == 0:
                        self.__thousand_offsets.append(self.__filehandle.tell())
                line = self.__filehandle.readline()
            self.__filehandle.seek(0)
            
        def __len__(self):
            return self.__lines
            
        def __getitem__(self, i):
            if i < 0 or i >= len(self):
                raise IndexError("Out of bounds")
            
            thousand_offset = self.__thousand_offsets[i // 1000]
            remainder = i % 1000
            
            self.__filehandle.seek(thousand_offset)
            while remainder > 0:
                self.__filehandle.readline()
                remainder -= 1
                
            name, url = [x.strip() for x in self.__filehandle.readline().strip().split("\t")]
            class_name = name.split("_")[0]
            
            pre_image = Image(name, class_name, url, None, None)

            submitted_task = self.__loader.submit(load_image, pre_image.url)
            def deferred_load():
                return submitted_task.result()
                                
            return Image(
                name, 
                class_name, 
                url, 
                load_annotation_for_image(pre_image),
                deferred_load
            )

        def close(self):
            self.__loader.shutdown()
            self.__filehandle.close()
            
    result = DelayedFileList(this_file_path / "fall11_urls.txt")
    return result


class TorchWrapper:
    def __init__(self, obj):
        self.obj = obj
        
    def __len__(self):
        return len(self.obj)

    def __getitem__(self, i):
        print(i)
        item = self.obj[i]
        
        image_data = None
        if item is not None:
            image_data = item.load()
        
        if image_data is None:
            return {
                "data": np.zeros((100, 100, 3), dtype=np.uint8),
                "seg": np.zeros((100, 100, 3), dtype=np.uint8),
                "fnames": np.uint8(i)
            }
        else:
            return {
                "data": image_data.astype(np.uint8),
                "seq": image_data.astype(np.uint8),
                "fnames": np.uint8(i)
            }
