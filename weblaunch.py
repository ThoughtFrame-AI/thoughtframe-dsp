# thoughtframe/main.py

import asyncio
import sys 

from tf_core.bootstrap import configure, thoughtframe
from tf_core.web.webserver import BaseWebServer

from thoughtframe.sensor.SensorMeshManager import SensorMeshManager
from thoughtframe.sensor.mesh_config import THOUGHTFRAME_APP_CONFIG
from thoughtframe.sensor.web.sensor_module import SensorModule


async def heartbeat():
    while True:
        print("heartbeat")
        await asyncio.sleep(5) 


        
async def main():
    ##setup TF itself
    configure(THOUGHTFRAME_APP_CONFIG)
    ##Add out own 
    thoughtframe.manager.register("sensormeshmanager",lambda: SensorMeshManager(thoughtframe.manager))
    thoughtframe.manager.register("SensorModule",lambda: SensorModule())

    connection_service = thoughtframe.get("connection")
   
    sensormanager  = thoughtframe.get("sensormeshmanager")
    ##sensormanager.start()
    
    web : BaseWebServer = thoughtframe.get("web")
    await web.start()

    
    await asyncio.Event().wait()
    
    
asyncio.run(main())