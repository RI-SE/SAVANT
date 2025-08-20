# SAVANT file formats

## SAVANT OpenLabel Subset
The tag file adheres to the ASAM OpenLabel format, but we only use a subset of the tags for SAVANT.
*Subject to change during development, this is an initial guess*

```json
{
  "openlabel": {
```

Only OpenLabel schema version is required in the metadata section. We use tagged_file to indicate the source video file, and annotator to initially add which version of the auto annotator is used or if it was annotated using the UI.
```json
    "metadata": { 
        "schema_version": "1.0.0",
        "tagged_file": "filename.mp4",
        "annotator": "SAVANT AutoAnno v0.1, SAVANT AnnoUI v0.1",
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
            "ontology_uid": "0"
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
            "ontology_uid": "0",
            "frame_intervals": [{ "frame_start": 5, "frame_end": 8 }]
        }
    },
```

Frames hold dynamic information, i.e. information that changes between frames. We use this for the bounding boxes of dynamic objects. The format allows for different types like 2d bounding boxes, polygons, and 3d boxes. 

We only use one type, which is rotated bounding box which is a rectangle with an angle (see image below). The x and y coordinates are to the center of the object and rotation is also calculated relative this center. We also add two custom fields with "vec", which is the name (or tool + version) of the annotator, and an associated confidence value.
```json
    "frames": {
      "0": {
        "objects": {
          "0": {
            "object_data": {
              "rbbox": [{ "name": "shape", "val": [x_px, y_px, width_px, height_px, alpha_rad] }],
              "vec": [
                { "name": "annotator", "val": [string, ...] },
                { "name": "confidence",   "val": [number, ...] }
              ]
            }
          }
        }
      },
      "1": {
        "objects": {
          "0": {
            "object_data": {
              "rbbox": [{ "name": "shape", "val": [400, 200, 10, 3, 0.24] }],
              "vec": [
                { "name": "annotator", "val": ["SAVANT AutoAnno v0.1", "thanh"] },
                { "name": "confidence",   "val": [0.87, 0.91] }
              ]
            }
          }
        }
      }
    } 
```

![RBBOX](rbbox.svg)

## SAVANT Ontology

This part is TBD. There is an Ontology provided by openlabel, called openlabel_ontology_scenario_tags.ttl, mainly for the scenario tags part of OpenLabel and not the . It does contain most of what we need in terms of objects, behaviours and ODD information, so maybe we should use it, but we must make sure it can be meaningfully converted to OSI and OpenScenario as well. It may be necessary to add something in this regard which means we need an own ontology anyway.

Also, if we use the ASAM one, again we should define a subset to be used within SAVANT. Which terms to be included (or excluded) can be defined when specifying an ontology in the ontologies section by using include and exclude sections, e.g.:
```json
    "ontologies": {
      "0": {
        "uri": "https://openlabel.asam.net/V1-0-0/ontologies/openlabel_ontology_scenario_tags.ttl",
        "boundary_list": ["motorway", "road"],
        "boundary_mode": "include"
      }
```
We can start with a fixed set of tags defineg e.g. in a vector in the UI. However, the architecture should be such that it will later be possible to read a set of tags from a file, i.e. the tags should not be hardcoded throughout the application.

### Base tags

| Tag | UID | Parent | Description |
|:---:|:---:|:---:|:---:|
| Tag | 0 |||
| DynamicObject | 1 | Tag ||
| StaticObject | 2 | Tag ||
| Action | 3 | Tag ||
| Map | 4 | Tag ||

### Dynamic object tags
_Y in the "Auto" column means this tag can be set by the auto annotator. Remaining tags are only available in the UI._
| Tag | UID | Parent | Auto | Description |
|:---:|:---:|:---:|:---:|:---:|
| RoadUser | 10 | DynamicObject | | |
| Vehicle | 11 | RoadUser | ||
| Car | 110 | Vehicle | Y ||
| Van | 111 | Vehicle | Y ||
| Truck | 112 | Vehicle | Y ||
| Trailer | 113 | Vehicle | Y ||
| Motorbike | 114 | Vehicle | Y ||
| Bicycle | 115 | Vehicle | ||
| Bus | 116 | Vehicle | Y ||
| Tram | 117 | Vehicle | ||
| Train | 118 | Vehicle | ||
| Caravan | 119 | Vehicle | ||
| StandupScooter | 120 | Vehicle | ||
| AgriculturalVehicle | 121 | Vehicle | ||
| ConstructionVehicle | 122 | Vehicle | ||
| EmergencyVehicle | 123 | Vehicle | ||
| SlowMovingVehicle | 124 | Vehicle | ||
| Human | 13 | RoadUser | ||
| Pedestrian | 130 | Human | ||
| WheelChairUser | 131 | ||
| Animal | 14 | RoadUser | ||
All dynamic objects are marked with the OpenLabel type rbbox - rotated bounding box, i.e. a rectangle with arbitrary rotation with parameters xywha.

### Static objects
| Tag | UID | Parent | Auto | Description |
|:---:|:---:|:---:|:---:|:---:|
| Aruca | 20 | StaticObject | Y ||

### Action tags
| Tag | UID | Parent | Auto | Description |
|:---:|:---:|:---:|:---:|:---:|
| Motion | 30 |Action |||
| TurnLeft | 300 | Motion |||
| TurnRight | 301 | Motion |||
| Cross | 302 | Motion |||
| CutIn | 303 | Motion |||
| CutOut | 304 | Motion |||
| Overtake | 305 | Motion |||
| Accelerate | 306 | Motion |||
| Decelerate | 307 | Motion |||
| LaneChangeRight | 308 | Motion |||
| LaneChangeLeft | 309 | Motion |||

### Map tags
_NOTE: THIS IS NOT FINALIZED
| Tag | UID | Parent | Auto | Description |
|:---:|:---:|:---:|:---:|:---:|
| Lane | 40 | Map | ||
| BikeLane | 41 | Map | ||
| Pavement | 42 | Map | ||
_We may need to be able to mark lanes to be able to create OpenDrive and to place objects relative to the lane. Still TBD how to handle this, e.g. direction of lane, how to mark lanes in crossing or roundabout etc. Lanes could be marked with 2Dpolyline (intead of rbbox like dynamic objects)._
