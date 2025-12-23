import asyncio

from thoughtframe.sensor import SensorManager
from thoughtframe.sensor.SensorProcessorManager import SensorProcessorManager
from thoughtframe.sensor.interface import AcousticSensor, AcousticPipeline, \
    AcousticChunkProcessor
from thoughtframe.sensor.mesh_config import MESH_CONFIG



class SensorMeshManager:
    def __init__(self, manager):
        self._manager = manager
        self._sensor_factories = {}
        self.register_sensor_types()
        self.pm = SensorProcessorManager()
        self.sm = SensorManager()
        
  
    def start(self):
        self.mesh = SensorMesh(self, MESH_CONFIG)
        self.mesh.build()
        self.mesh.run()

class SensorNode:
    
    def __init__(self, sensor: AcousticSensor, pipeline : AcousticPipeline):
        self.sensor = sensor
        self.pipeline = pipeline
    

class SensorMesh:
      
    def __init__(self, meshmanager, MESH_CONFIG):
        self.meshmanager = meshmanager;
        self.nodes =[]
        self.config = MESH_CONFIG
        self.tasks = []
        
    def build(self):
        for item in self.config['sensors']:
            print(item)
            sensor:AcousticSensor = self.meshmanager.sm.createSensor(item)
            pipelineconfig =item['pipeline']
            pipeline:AcousticPipeline = AcousticPipeline()
            for element in pipelineconfig:
                chunkprocessor: AcousticChunkProcessor = self.meshmanager.pm.createProcessor(element, sensor)
                pipeline.addChunkProcessor(chunkprocessor)
            node = SensorNode(sensor, pipeline)
            self.nodes.append(node)
                
    
    async def _run_node(self, node: SensorNode):
        ##TODO:  rewire this 
        ##worker = AcousticMLWorker(fs=node.sensor.fs)
        pipeline : AcousticPipeline= node.pipeline
        async for chunk in node.sensor.stream():
            analysis = pipeline.execute(chunk,node)
            ##print(analysis)

    
    def run(self):
        for sensornode in self.nodes:
            task = asyncio.create_task(self._run_node(sensornode))
            self.tasks.append(task)
        
        
            
