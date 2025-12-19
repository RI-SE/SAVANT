# SAVANT Ontology

This directory contains the formal ontology for the SAVANT video annotation toolkit.

## Overview

The SAVANT ontology defines classes for objects, behaviours, and relations in traffic scenarios, designed for UAV-based video annotation and object detection.

**Namespace URI:** `http://github.com./RI-SE/SAVANT/ontology#`
**Prefix:** `savant:`
**Version:** 1.3.1
**Format:** Turtle (RDF)

## Files

- **`savant.ttl`** - Current ontology definition in Turtle format (v1.3.1)
- **`README.md`** - This documentation file

## UID Ranges

Classes are assigned unique identifiers (UIDs) in specific ranges:

| Range | Category | Description |
|-------|----------|-------------|
| 0-99 | Dynamic Objects | Moving objects (vehicles, humans, animals) |
| 200-299 | Static Objects | Fixed objects (markers, infrastructure) |
| 300-399 | Behaviours | Actions and motions |
| 400-499 | Relations | Object relationships |

Dynamic object UIDs (0-99) align with YOLO class indexes for training.

## Classes

### Dynamic Objects (uid 0-24)

Road users and moving entities detected in video.

| UID | Class | Label | Parent |
|-----|-------|-------|--------|
| 0 | DynamicObject | DynamicObject | - |
| 1 | RoadUser | RoadUser | DynamicObject |
| 2 | RoadUserVehicle | Vehicle | RoadUser |
| 3 | VehicleCar | Car | Vehicle |
| 4 | VehicleVan | Van | Vehicle |
| 5 | VehicleTruck | Truck | Vehicle |
| 6 | VehicleTrailer | Trailer | Vehicle |
| 7 | VehicleMotorbike | Motorbike | Vehicle |
| 8 | VehicleBicycle | Bicycle | Vehicle |
| 9 | VehicleBus | Bus | Vehicle |
| 10 | VehicleTram | Tram | Vehicle |
| 11 | VehicleTrain | Train | Vehicle |
| 12 | VehicleCaravan | Caravan | Vehicle |
| 13 | VehicleStandupScooter | StandupScooter | Vehicle |
| 14 | VehicleAgriculturalVehicle | AgriculturalVehicle | Vehicle |
| 15 | VehicleConstructionVehicle | ConstructionVehicle | Vehicle |
| 16 | VehicleEmergencyVehicle | EmergencyVehicle | Vehicle |
| 17 | VehicleAmbulance | Ambulance | Vehicle |
| 18 | VehicleFire | Fire | Vehicle |
| 19 | VehiclePolice | Police | Vehicle |
| 20 | VehicleSlowMovingVehicle | SlowMovingVehicle | Vehicle |
| 21 | RoadUserHuman | Human | RoadUser |
| 22 | HumanPedestrian | Pedestrian | Human |
| 23 | HumanWheelChairUser | WheelChairUser | Human |
| 24 | Animal | Animal | RoadUser |

### Static Objects (uid 200+)

Fixed objects used as reference points or infrastructure.

| UID | Class | Label | Parent |
|-----|-------|-------|--------|
| 200 | StaticObject | StaticObject | - |
| 210 | Marker | Marker | StaticObject |
| 211 | MarkerAruco | Aruco | Marker |
| 212 | MarkerVisualMarker | VisualMarker | Marker |
| 220 | RoadInfrastructure | RoadInfrastructure | StaticObject |

### Behaviours (uid 300+)

Actions and motion patterns for annotating object behaviour.

| UID | Class | Label | Parent |
|-----|-------|-------|--------|
| 300 | Behaviour | Behaviour | - |
| 310 | BehaviourMotion | Motion | Behaviour |
| 311 | MotionTurnLeft | TurnLeft | Motion |
| 312 | MotionTurnRight | TurnRight | Motion |
| 313 | MotionCross | Cross | Motion |
| 314 | MotionCutIn | CutIn | Motion |
| 315 | MotionCutOut | CutOut | Motion |
| 316 | MotionOvertake | Overtake | Motion |
| 317 | MotionAccelerate | Accelerate | Motion |
| 318 | MotionDecelerate | Decelerate | Motion |
| 319 | MotionLaneChangeRight | LaneChangeRight | Motion |
| 320 | MotionLaneChangeLeft | LaneChangeLeft | Motion |

### Relations (uid 400+)

Relationships between objects.

| UID | Class | Label | Parent |
|-----|-------|-------|--------|
| 400 | Relation | Relation | - |
| 410 | RelationTowing | towed-by | Relation |
| 420 | RelationCarrying | carried-by | Relation |

## Properties

### uid

**Definition:** A unique identifier assigned to a class for YOLO training alignment and OpenLabel export.

**Type:** `owl:AnnotationProperty`
**Range:** `xsd:integer`

## Usage with SAVANT Tools

### markit

The ontology is used by markit to map YOLO class IDs to semantic labels:

```bash
markit --input video.mp4 --output_json output.json \
       --ontology ontology/savant.ttl
```

### train-yolo-obb

Training uses UIDs 0-24 as YOLO class indexes:

```bash
train-yolo-obb --data dataset.yaml --epochs 50
```

## Validation

Validate the ontology file with RDF tools:

```bash
# Using rapper (raptor2-utils)
rapper -i turtle -o ntriples savant.ttl > /dev/null

# Using riot (Apache Jena)
riot --validate savant.ttl
```

## Extending the Ontology

To add new classes, follow the UID conventions:

```turtle
# New vehicle type (use next available UID in 0-99 range)
:VehicleScooter a rdfs:Class ;
    rdfs:subClassOf :RoadUserVehicle ;
    rdfs:label "Scooter" ;
    :uid 25 .

# New static object (use next available UID in 200-299 range)
:TrafficSign a rdfs:Class ;
    rdfs:subClassOf :RoadInfrastructure ;
    rdfs:label "TrafficSign" ;
    :uid 221 .
```

## License

SAVANT is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

## References

- [SAVANT Repository](https://github.com/RI-SE/SAVANT)
- [OpenLABEL Specification](https://www.asam.net/standards/detail/openlabel/)
- [Ultralytics YOLO](https://docs.ultralytics.com/)
