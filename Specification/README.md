# SAVANT file formats

The tag file adheres to the ASAM OpenLabel format, but we only use a subset of the tags for SAVANT.
*Subject to change during development, this is an initial guess*

```json
{
  "openlabel": {
```

Only OpenLabel schema version is required in the metadata section. We use tagged_file to indicate the source video file, and annotator to initially add which version of the auto annotator is used or if it was annotated using the UI.
```json
{
    "metadata": { 
        "schema_version": "1.0.0",
        "tagged_file": "filename.mp4",
        "annotator": "SAVANT autoanno v0.1",
    },
```

Ontologies is a definition of types. We may use our own, or the one used for openlabel scenario tagging, or both. But we need to define types for the objects we want to tag, and the actions (e.g. overtake, pedestrian crossing) we want to be able to tag in the UI. Every ontology has a unique uid which can be an unsigned integer sequence ("0" in the example) or a uuid (which we don't use).
```json
    "ontologies" : {
        "0" : "https://savant.ri.se/savant_ontology_1.0.0.ttl"
    },
```
Objects carry static information about objects in the project, i.e. the type and name of objects appearing in the sequence can be held here. It may be static information about dynamic objects (where the dynamic information is in the frame tags, see below) or static information about static objects (like a sign) which only need to appear once.

frame_intervals in not required. ontology_uid is not required, but points to which ontology defines the type (there may be several ontologies used in the project). Every object has a uniquq uid. 
```json
    "objects": {
        "0": {
            "name": "Car-0",
            "type": "Car",
            "ontology_uid" : "0",
            "frame_intervals": [{ "frame_start": 0, "frame_end": 10 }]
        } ,
        "1": {
            "name": "Person-1",
            "type": "Pedestrian",
            "ontology_uid": 0
        }
     },
```

Actions define semantically meaningful acts which may occur over frame intervals, e.g. an overtake.
There are also events (similar to actions but instantaneous) and context (e.g. scene properties or weather conditions). We don't use them, at least not yet. 
```json
    "actions" : {
        "0": {
            "name": "Action-0",
            "type": "Overtake",
            "ontology_uid": 0,
            "frame_intervals": [{ "frame_start": 5, "frame_end": 8 }]
        }
    },
```

Frames hold dynamic information, i.e. information that changes between frames. We use this for the bounding boxes of dynamic objects. The format allows for different types like 2d bounding boxes, polygons, and 3d boxes. We only use one type, which is rotated bounding box which is a rectangle with an angle (see image below). The x and y coordinates are to the center of the object and rotation is also calculated relative this center.
```json
    "frames": {
      "0": {
        "objects": {
          "0": {
            "object_data": {
              "rbbox": [{ "name": "shape", "val": [x_px, y_px, width_px, height_px, alpha_rad] }]
            }
          }
        }
      },
      "1": {
        "objects": {
          "0": {
            "object_data": {
              "rbbox": [{ "name": "shape", "val": [400, 200, 10, 3, 0.24] }]
            }
          }
        }
      }
    }
  }
}
```

![RBBOX](rbbox.svg)
