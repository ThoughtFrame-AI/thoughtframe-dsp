# thoughtframe/main.py

import asyncio
import sys 

from tf_core.bootstrap import configure, thoughtframe
from tf_core.web.webserver import BaseWebServer

from thoughtframe.sensor.SensorMeshManager import SensorMeshManager
from thoughtframe.sensor.mesh_config import THOUGHTFRAME_APP_CONFIG


async def heartbeat():
    while True:
        print("heartbeat")
        await asyncio.sleep(5) 


        
async def main():
    ##setup TF itself
    configure(THOUGHTFRAME_APP_CONFIG)
    ##Add out own 
    thoughtframe.manager.register("sensormeshmanager",lambda: SensorMeshManager(thoughtframe.manager))
    
    connection_service = thoughtframe.get("connection")
   
    sensormanager  = thoughtframe.get("sensormeshmanager")
    sensormanager.start()
    
    web : BaseWebServer = thoughtframe.get("web")
    

    
    await asyncio.Event().wait()
    
    
asyncio.run(main())