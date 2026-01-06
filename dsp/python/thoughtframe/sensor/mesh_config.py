PIPELINE_BASE = [
    {"op": "debug"},
    {"op": "spectral_features"},
    {"op": "isolation_forest", "threshold": -0.1},
    {"op": "temporal_context", "time": "1h"},
    {"op": "if_window_isolator",
     "threshold": -0.15,
     "min_duration": "1m"},
    {"op": "impulse_isolator",
        "threshold_mult": 10.0,
        "window_sec": 5.0,
        "min_impulses": 5},
   
    # {"op": "guard",
    #  "name": "impulses_high",
    #  "band": [600, 3000],
    #  "pipeline": [
    #      {"op": "impulse_isolator",
    #       "threshold_mult": 10.0,
    #       "window_sec": 5.0,
    #       "min_impulses": 5}
    #  ]},
    # {"op": "guard",
    #  "name": "impulses_low",
    #  "band": [20, 250],
    #  "pipeline": [
    #      {"op": "impulse_isolator",
    #       "threshold_mult": 10.0,
    #       "window_sec": 5.0,
    #       "min_impulses": 5}
    #  ]},
    {"op": "ring_buffer", "seconds": 20},
    {"op": "telemetry"}
]


MESH_CONFIG = {
    "sources": [
        {
            "id": "mic1",
            "type": "ffmpeg",
            "cmd": [
                "ffmpeg",
                "-i", "samples/OneDayHydrophone.flac",
                "-ac", "1",
                "-ar", "48000",
                "-f", "f32le",
                "-"
            ],
            "fs": 48000,
            "chunk_size": 4096
        }
    ],

    "arrays": [
        {
            "id": "towed_array",
            "inputs": ["mic1"],
            "geometry": {
                "spacing_m": 0.5,
                "sound_speed": 1500
            },
            "beams": [
                {"id": "beam_-30", "angle": -30},
                {"id": "beam_0",   "angle": 0},
                {"id": "beam_30",  "angle": 30}
            ]
        }
    ],

   
    "sensors": [
        # {
        #     "id": "beam_-30",
        #     "type": "beam",
        #     "array": "towed_array",
        #     "pipeline": PIPELINE_BASE
        # }
        # ,
        {
            "id": "beam_0",
            "type": "beam",
            "array": "towed_array",
            "pipeline": PIPELINE_BASE
        },
        # {
        #     "id": "beam_30",
        #     "type": "beam",
        #     "array": "towed_array",
        #     "pipeline": PIPELINE_BASE
        # }
    ]
}



THOUGHTFRAME_CONFIG  = {
    "root":".",
    "samples":"/audio"
}




NETWORK_CONFIG  = {
    "baseurl":"https://localhost:8080",
    "websocket":"ws://localhost:8080/entermedia/services/websocket/org/thoughtframe/websocket/BenchConnection?sessionid={tabID}",
    "apphome":"/thoughtframe"
}


THOUGHTFRAME_APP_CONFIG = {
    "mesh": MESH_CONFIG,
    "paths": THOUGHTFRAME_CONFIG,
    "network": NETWORK_CONFIG,
}

