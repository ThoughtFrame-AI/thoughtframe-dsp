import asyncio
import json
from pathlib import Path

from thoughtframe.sensor.SensorManager import SensorManager
from thoughtframe.sensor.SensorProcessorManager import SensorProcessorManager
from thoughtframe.sensor.array_coordinator import ArrayCoordinator
from thoughtframe.sensor.interface import AcousticSensor, AcousticPipeline   
from thoughtframe.sensor.mesh_config import MESH_CONFIG
from thoughtframe.sensor.sensors.QueueSensor import QueueSensor


class SensorMeshManager:
    def __init__(self, manager):
        self._manager = manager
        self._sensor_factories = {}
        self.pm = SensorProcessorManager()
        self.sm = SensorManager()
        self.mesh = None
        self.running = False
  
    def start(self, request=None):
        if self.running:
            return           
            
        if request and "mesh" in request:
            cfg = request["mesh"]
            source = "inline"
        else:
            cfg_path = (
                Path(__file__).parents[1] / "config" / "mesh.json"
            )
            with cfg_path.open() as f:
                cfg = json.load(f)
            source = str(cfg_path)

        self.mesh = SensorMesh(self, cfg)
        self.mesh.build()

        asyncio.get_running_loop().call_soon(self.mesh.run)

        return {
            "status": "started",
            "config_source": source
        }

class SensorNode:
    
    def __init__(self, sensor: AcousticSensor, pipeline : AcousticPipeline):
        self.sensor = sensor
        self.pipeline = pipeline
        self.chunk_index = 0
        self.t_sec = 0

    

class SensorMesh:
      
    def __init__(self, meshmanager, config):
        self.mesh = meshmanager;
        self.nodes =[]
        self.config = config
        self.tasks = []
        self.source_tasks = []
        self.sensor_registry = {}      # id -> AcousticSensor
        self.array_registry = {}       # id -> ArrayCoordinator
        
        
    def build(self):
        self._build_sources()
        self._build_arrays()
        self._build_pipeline_nodes()

    def _build_arrays(self):
        for cfg in self.config.get("arrays", []):
            inputs = [self.sensor_registry[sid] for sid in cfg["inputs"]]
    
            array = ArrayCoordinator(cfg, inputs)
            self.array_registry[cfg["id"]] = array
    
            # Register beam sensors produced by this array
            for beam in cfg["beams"]:
                beam_sensor = array.get_beam_sensor(beam["id"])
                self.sensor_registry[beam["id"]] = beam_sensor
    
            # Schedule the array task
            self.source_tasks.append(array)
    
    def _build_sources(self):
        for cfg in self.config.get("sources", []):
            sensor = self.mesh.sm.createSensor(cfg)
            self.sensor_registry[cfg["id"]] = sensor

    def _build_pipeline_nodes(self):
        for cfg in self.config.get("sensors", []):
            sensor_id = cfg["id"]
    
            sensor = self.sensor_registry[sensor_id]
    
            pipeline = AcousticPipeline()
            for element in cfg["pipeline"]:
                proc = self.mesh.pm.createProcessor(element, sensor)
                pipeline.addChunkProcessor(proc)
    
            node = SensorNode(sensor, pipeline)
            self.nodes.append(node)            
    
    async def _run_node(self, node: SensorNode):
        
        dt = node.sensor.chunk_size / node.sensor.fs
        pipeline : AcousticPipeline= node.pipeline
        async for chunk in node.sensor.stream():
            node.chunk_index += 1
            node.t_sec = node.chunk_index * dt
            pipeline.execute(chunk, node)

            ##print(analysis.metadata["t_sec"])

    
    def run(self):
        for src in self.source_tasks:
            self.tasks.append(asyncio.create_task(src.run()))
    
        for node in self.nodes:
            self.tasks.append(asyncio.create_task(self._run_node(node)))
        print("Running Sensor Mesh")
        
            
