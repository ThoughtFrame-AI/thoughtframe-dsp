import asyncio
import json
from pathlib import Path

from thoughtframe.perception.PerceptionManager import PerceptionManager
from thoughtframe.perception.PerceptionProcessorManager import PerceptionProcessorManager
from thoughtframe.perception.interface import TensorSource, PerceptionPipeline


class PerceptionMeshManager:
    def __init__(self, manager):
        self._manager = manager
        self.sources = {}
        self.pm = PerceptionProcessorManager()
        self.sm = PerceptionManager()
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
                Path(__file__).parents[1] /"perception"/ "config" / "mesh.json"
            )
            with cfg_path.open() as f:
                cfg = json.load(f)
            source = str(cfg_path)

        self.mesh = PerceptionMesh(self, cfg)
        self.mesh.build()

        asyncio.get_running_loop().call_soon(self.mesh.run)
        self.running =  True
        return {
            "status": "started",
            "config_source": source
        }


class PerceptionNode:
    def __init__(self, source: TensorSource, pipeline: PerceptionPipeline):
        self.source = source
        self.pipeline = pipeline
        self.index = 0


class PerceptionMesh:
    def __init__(self, meshmanager, config):
        self.mesh = meshmanager
        self.nodes = []
        self.config = config
        self.tasks = []
        self.source_tasks = []
        self.sources = {}   # id -> TensorSource

    def build(self):
        self._build_sources()
        self._build_pipeline_nodes()

    def _build_sources(self):
        for cfg in self.config.get("sources", []):
            source = self.mesh.sm.createSource(cfg)
            self.sources[cfg["id"]] = source

    def _build_pipeline_nodes(self):
        for cfg in self.config.get("sources", []):
            source_id = cfg["id"]

            source = self.sources[source_id]

            pipeline = PerceptionPipeline()
            for element in cfg.get("pipeline", []):
                proc = self.mesh.pm.createProcessor(element, source)
                pipeline.addProcessor(proc)

            node = PerceptionNode(source, pipeline)
            self.nodes.append(node)

    async def _run_node(self, node: PerceptionNode):
        print("Starting node for source", node.source.source_id)
    
        async for item in node.source.stream():
            try:
                node.pipeline.execute(item, node)
                node.index += 1
            except Exception as e:
                print(
                    f"[Perception] Pipeline error on source "
                    f"{node.source.source_id}: {e}",
                    flush=True,
                )
                # IMPORTANT: continue so the task stays alive


    def run(self):
        for node in self.nodes:
            task = asyncio.create_task(self._run_node(node))
            self.tasks.append(task)
    
        print("Perception mesh running")


    