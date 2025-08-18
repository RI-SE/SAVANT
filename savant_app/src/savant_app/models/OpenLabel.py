from pydantic import BaseModel, conint, confloat, model_serializer, model_validator, Field
from typing import Dict, List, Literal, Union, Optional

class RotatedBBox(BaseModel):
    """Rotated bounding box coordinates (x_center, y_center, width, height, rotation)"""
    x_center: confloat(ge=0)
    y_center: confloat(ge=0)
    width: confloat(gt=0)
    height: confloat(gt=0)
    rotation: float  # radians

    @model_serializer
    def serialize(self) -> list:
        return [self.x_center, self.y_center, self.width, self.height, self.rotation]

    @model_validator(mode='before')
    @classmethod
    def validate_deserialize(cls, data):
        if isinstance(data, list):
            if len(data) != 5:
                raise ValueError("RotatedBBox requires exactly 5 elements for deserialization")
            return {
                'x_center': data[0],
                'y_center': data[1],
                'width': data[2],
                'height': data[3],
                'rotation': data[4]
            }
        return data

class GeometryData(BaseModel):
    """Represents geometric data for rotated bounding boxes"""
    name: Literal['shape']
    val: RotatedBBox

class ConfidenceData(BaseModel):
    """Contains confidence score for detection"""
    name: Literal['confidence']
    val: List[confloat(ge=0, le=1)]  # List of confidence scores

class AnnotatorData(BaseModel):
    """Contains annotator information"""
    name: Literal['annotator']
    val: List[str]

class ObjectData(BaseModel):
    """Container for object's geometric and confidence data"""
    rbbox: List[GeometryData]
    vec: List[Union[ConfidenceData, AnnotatorData]]

class FrameLevelObject(BaseModel):
    """Represents an object's data within a specific frame"""
    object_data: ObjectData

class FrameObjects(BaseModel):
    """Represents all objects within a specific frame"""
    objects: Dict[str, FrameLevelObject]

class FrameInterval(BaseModel):
    """Represents the frames in which the object exists"""
    frame_start: conint(ge=0)
    frame_end: conint(ge=0)

class ObjectMetadata(BaseModel):
    """Metadata for tracked objects across all frames"""
    name: str
    type: str
    ontology_uid: Optional[Union[int, str]] = None
    frame_intervals: Optional[List[FrameInterval]] = None

class OpenLabelMetadata(BaseModel):
    """Top-level metadata for the OpenLabel annotation file"""
    schema_version: str
    tagged_file: Optional[str] = None
    annotator: Optional[str] = None

class ActionMetadata(BaseModel):
    """Action metadata"""
    name: str
    type: str
    ontology_uid: Optional[Union[int, str]] = None
    frame_intervals: Optional[List[FrameInterval]] = None

class OntologyDetails(BaseModel):
    """ 
    Ontology details which are not yet in use, but are ready for when needed.
    Refer to the following section in the readme: https://github.com/fwrise/SAVANT/tree/main/Specification#savant-ontology
    """
    uri: str
    boundary_list: Optional[List[str]] = None
    boundary_mode: Optional[Literal["include", "exclude"]] = None

class OpenLabel(BaseModel):
    """Main model representing the complete OpenLabel structure"""
    metadata: OpenLabelMetadata
    ontologies: Dict[str, Union[str, OntologyDetails]]
    objects: Dict[str, ObjectMetadata]
    actions: Optional[Dict[str, ActionMetadata]] = None  # Made optional as they are not being used yet according to the spec.
    frames: Dict[str, FrameObjects]

    def model_dump(self, *args, **kwargs) -> dict:
        """"
        This overrides Pydantic's default model_dump, such that
        we exclude fields with a None value. 
        """
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(*args, **kwargs)
